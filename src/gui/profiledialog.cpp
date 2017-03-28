#include "profiledialog.h"
#include "ui_profiledialog.h"
#include "session.h"

ProfileDialog::ProfileDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProfileDialog),
    session(s)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);

    connect(&session, SIGNAL(actionLogged(QString,QString,QString,int)), this, SLOT(updateCombo()));
    connect(ui->cbSelection, SIGNAL(currentIndexChanged(int)), this, SLOT(updateRange()));
    connect(ui->xRange, SIGNAL(stateChanged(int)), this, SLOT(updateRange()));

    connect(this, SIGNAL(profile()), &session.profiler(), SLOT(profile()));
    connect(&session.profiler(), SIGNAL(profileComplete(int)), this, SLOT(profileComplete(int)));
    connect(&session.profiler(), SIGNAL(done()), this, SLOT(done()));

    updateCombo();
}

ProfileDialog::~ProfileDialog()
{
    delete ui;
}

void ProfileDialog::updateCombo()
{
    int currentIdx = ui->cbSelection->currentIndex();
    QSize currentData = ui->cbSelection->currentData().toSize();
    ui->cbSelection->clear();
    for ( size_t i = 0; i < session.wavegen().archives().size(); i++ )
        ui->cbSelection->addItem(session.wavegen().prettyName(i), QSize(0, i));
    for ( size_t i = 0; i < session.wavegenselector().selections().size(); i++ )
        ui->cbSelection->addItem(session.wavegenselector().prettyName(i), QSize(1, i));

    if ( currentIdx < 0 )
        return;

    ui->cbSelection->setCurrentIndex(currentData.width() * session.wavegen().archives().size() + currentData.height());
}

void ProfileDialog::updateRange()
{
    if ( ui->cbSelection->currentIndex() < 0 )
        return;

    const Wavegen::Archive *archive;
    QSize collection = ui->cbSelection->currentData().toSize();
    if ( collection.width() ) {
        archive =& session.wavegenselector().selections().at(collection.height()).archive();
    } else {
        archive =& session.wavegen().archives().at(collection.height());
    }

    double min = session.project.model().adjustableParams[archive->param].min;
    double max = session.project.model().adjustableParams[archive->param].max;
    ui->sbMin->setRange(min, max);
    ui->sbMax->setRange(min, max);
    ui->sbMin->setEnabled(ui->xRange->isChecked());
    ui->sbMax->setEnabled(ui->xRange->isChecked());
    if ( !ui->xRange->isChecked() ) {
        ui->sbMin->setValue(min);
        ui->sbMax->setValue(max);
    }
}

void ProfileDialog::on_btnStart_clicked()
{
    if ( ui->cbSelection->currentIndex() < 0 )
        return;

    int param;
    std::vector<Stimulation> stim;

    QSize collection = ui->cbSelection->currentData().toSize();
    if ( collection.width() ) {
        const WavegenSelection &sel = session.wavegenselector().selections().at(collection.height());
        stim.reserve(sel.size());
        std::vector<size_t> idx(sel.ranges.size());
        for ( size_t i = 0; i < sel.size(); i++ ) {
            for ( int j = sel.ranges.size() - 1; j >= 0; j-- ) {
                if ( ++idx[j] % sel.width(j) == 0 )
                    idx[j] = 0;
                else
                    break;
            }
            bool ok;
            auto it = sel.data_relative(idx, &ok);
            if ( ok )
                stim.push_back(it->wave);
        }
        param = sel.archive().param;
    } else {
        const Wavegen::Archive &archive = session.wavegen().archives().at(collection.height());
        stim.reserve(archive.elites.size());
        for ( MAPElite const& e : archive.elites )
            stim.push_back(e.wave);
        param = archive.param;
    }

    session.profiler().setStimulations(stim);
    numStimulations = stim.size();


    std::vector<ErrorProfiler::Permutation> perm(session.project.model().adjustableParams.size());
    perm[param].n = 0;
    if ( ui->xRange->isChecked() ) {
        perm[param].min = ui->sbMin->value();
        perm[param].max = ui->sbMax->value();
    } else {
        perm[param].min = session.project.model().adjustableParams[param].min;
        perm[param].max = session.project.model().adjustableParams[param].max;
    }
    session.profiler().setPermutations(perm);

    ui->btnStart->setEnabled(false);
    ui->btnAbort->setEnabled(true);

    ui->log->addItem(QString("Profiling %1...").arg(ui->cbSelection->currentText()));
    ui->log->addItem("");
    ui->log->scrollToBottom();

    emit profile();
}

void ProfileDialog::on_btnAbort_clicked()
{
    session.profiler().abort();
}

void ProfileDialog::profileComplete(int index)
{
    QListWidgetItem *item = ui->log->item(ui->log->count()-1);
    item->setText(QString("%1/%2 ...").arg(index+1).arg(numStimulations));
}

void ProfileDialog::done()
{
    ui->btnStart->setEnabled(true);
    ui->btnAbort->setEnabled(false);
    ui->log->addItem("Done.");
    ui->log->scrollToBottom();
}
