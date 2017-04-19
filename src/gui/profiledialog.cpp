#include "profiledialog.h"
#include "ui_profiledialog.h"
#include "session.h"
#include <QSpinBox>
#include <QDoubleSpinBox>
#include "wavesource.h"

ProfileDialog::ProfileDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProfileDialog),
    session(s)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);

    connect(&session, SIGNAL(actionLogged(QString,QString,QString,int)), this, SLOT(updateCombo()));

    connect(this, SIGNAL(generate()), &session.profiler(), SLOT(generate()));
    connect(&session.profiler(), SIGNAL(progress(int,int)), this, SLOT(profileProgress(int,int)));
    connect(&session.profiler(), SIGNAL(done()), this, SLOT(done()));
    connect(&session.profiler(), SIGNAL(didAbort()), this, SLOT(aborted()));

    QStringList labels;
    int rows = session.project.model().adjustableParams.size();
    ui->table->setRowCount(rows);
    ns.resize(rows);
    mins.resize(rows);
    maxes.resize(rows);
    int i = 0;
    for ( AdjustableParam const& p : session.project.model().adjustableParams ) {
        labels << QString::fromStdString(p.name);

        QSpinBox *n = new QSpinBox();
        n->setRange(0, session.project.expNumCandidates());
        n->setSingleStep(100);
        ui->table->setCellWidget(i, 0, n);
        ns[i] = n;

        int decimals = 4 - log10(fabs(p.min == 0 ? p.initial : p.min));
        QDoubleSpinBox *min = new QDoubleSpinBox();
        min->setDecimals(decimals);
        min->setRange(p.min, p.max);
        min->setValue(p.min);
        ui->table->setCellWidget(i, 1, min);
        mins[i] = min;

        QDoubleSpinBox *max = new QDoubleSpinBox();
        max->setDecimals(decimals);
        max->setRange(p.min, p.max);
        max->setValue(p.max);
        ui->table->setCellWidget(i, 2, max);
        maxes[i] = max;

        connect(n, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=](int v) {
            if ( v == 0 ) {
                min->setValue(p.min);
                max->setValue(p.max);
            } else if ( v == 1 ) {
                min->setValue(p.initial);
                ui->table->model()->setData(ui->table->model()->index(i, 1), true, Qt::UserRole); // Set "clean" flag
            }
            min->setEnabled(v > 0);
            max->setEnabled(v > 1);
        });
        connect(min, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [=](double) {
            ui->table->model()->setData(ui->table->model()->index(i, 1), false, Qt::UserRole); // Set "dirty" flag
        });

        n->setValue(1);

        ++i;
    }
    ui->table->setVerticalHeaderLabels(labels);
    ui->table->setColumnWidth(0, 50);

    updateCombo();
}

ProfileDialog::~ProfileDialog()
{
    delete ui;
}

void ProfileDialog::updateCombo()
{
    int currentIdx = ui->cbSelection->currentIndex();
    WaveSource currentData = ui->cbSelection->currentData().value<WaveSource>();
    ui->cbSelection->clear();
    for ( size_t i = 0; i < session.wavegen().archives().size(); i++ ) {
        WaveSource src(session, WaveSource::Archive, i);
        ui->cbSelection->addItem(src.prettyName(), QVariant::fromValue(src));
    }
    for ( size_t i = 0; i < session.wavegenselector().selections().size(); i++ ) {
        WaveSource src(session, WaveSource::Selection, i);
        ui->cbSelection->addItem(src.prettyName(), QVariant::fromValue(src));
    }

    if ( currentIdx < 0 )
        return;

    ui->cbSelection->setCurrentIndex(currentData.index());
}

void ProfileDialog::on_btnStart_clicked()
{
    if ( ui->cbSelection->currentIndex() < 0 )
        return;

    std::vector<Stimulation> stim;

    WaveSource src = ui->cbSelection->currentData().value<WaveSource>();
    if ( src.type == WaveSource::Selection ) {
        const WavegenSelection &sel = *src.selection();
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
    } else {
        stim.reserve(src.archive().elites.size());
        for ( MAPElite const& e : src.archive().elites )
            stim.push_back(e.wave);
    }

    ErrorProfile profile(session);
    profile.setStimulations(std::move(stim));

    for ( size_t i = 0; i < session.project.model().adjustableParams.size(); i++ ) {
        ErrorProfile::Permutation perm;
        perm.n = ns[i]->value();
        if ( perm.n == 1 && !ui->table->model()->data(ui->table->model()->index(i, 1), Qt::UserRole).toBool() ) {
            perm.fixed = true;
            perm.value = mins[i]->value();
        }
        perm.min = mins[i]->value();
        perm.max = maxes[i]->value();
        profile.setPermutation(i, perm);
    }

    ui->log->addItem(QString("Profiling %1...").arg(ui->cbSelection->currentText()));
    ui->log->addItem("");
    ui->log->scrollToBottom();

    session.profiler().queueProfile(std::move(profile));
    emit generate();
}

void ProfileDialog::on_btnAbort_clicked()
{
    session.profiler().abort();
}

void ProfileDialog::profileProgress(int nth, int total)
{
    QListWidgetItem *item = ui->log->item(ui->log->count()-1);
    item->setText(QString("%1/%2 ...").arg(nth).arg(total));
}

void ProfileDialog::done()
{
    ui->log->addItem("Done.");
    ui->log->scrollToBottom();
}

void ProfileDialog::aborted()
{
    ui->log->addItem("Aborted.");
    ui->log->scrollToBottom();
}
