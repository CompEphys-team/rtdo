#include "profiledialog.h"
#include "ui_profiledialog.h"

ProfileDialog::ProfileDialog(ExperimentLibrary &lib, QThread *thread, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProfileDialog),
    thread(thread),
    lib(lib),
    profiler(lib),
    selections(nullptr)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);

    connect(this, SIGNAL(profile()), &profiler, SLOT(profile()));
    connect(&profiler, SIGNAL(profileComplete(int)), this, SLOT(profileComplete(int)));
    connect(&profiler, SIGNAL(done()), this, SLOT(done()));

    profiler.moveToThread(thread);
}

ProfileDialog::~ProfileDialog()
{
    delete ui;
}

void ProfileDialog::selectionsChanged(WavegenDialog *dlg)
{
    selections =& dlg->selections;
    ui->cbSelection->clear();
    for ( WavegenDialog::Selection const& sel : *selections )
        ui->cbSelection->addItem(dlg->name(sel));
    ui->btnStart->setEnabled(selections->size());
}

void ProfileDialog::on_btnStart_clicked()
{
    WavegenDialog::Selection const& sel = selections->at(ui->cbSelection->currentIndex());
    std::vector<Stimulation> stim(sel.elites.size());
    for ( size_t i = 0; i < stim.size(); i++ ) {
        stim[i] = sel.elites[i].wave;
    }
    profiler.setStimulations(stim);

    std::vector<ErrorProfiler::Permutation> perm(lib.model.adjustableParams.size());
    perm[sel.param].n = 0; // Use full range for target param (see Permutation defaults)
    profiler.setPermutations(perm);

    ui->btnStart->setEnabled(false);
    ui->btnAbort->setEnabled(true);

    ui->log->addItem(QString("Profiling %1...").arg(ui->cbSelection->currentText()));
    QListWidgetItem *item = new QListWidgetItem("");
    ui->log->addItem(item);
    item->setData(Qt::UserRole, int(stim.size()));
    ui->log->scrollToBottom();

    emit profile();
}

void ProfileDialog::on_btnAbort_clicked()
{
    profiler.abort();
}

void ProfileDialog::profileComplete(int index)
{
    QListWidgetItem *item = ui->log->item(ui->log->count()-1);
    item->setText(QString("%1/%2 ...").arg(index+1).arg(item->data(Qt::UserRole).toInt()));
}

void ProfileDialog::done()
{
    ui->btnStart->setEnabled(true);
    ui->btnAbort->setEnabled(false);
    ui->log->addItem("Done.");
    ui->log->scrollToBottom();
}
