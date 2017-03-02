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

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectPlottables | QCP::iSelectAxes);
    connect(ui->plot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->plot->selectedAxes();
        if ( axes.isEmpty() )
           axes = ui->plot->axisRect()->axes();
        ui->plot->axisRect()->setRangeZoomAxes(axes);
        ui->plot->axisRect()->setRangeDragAxes(axes);

    });
    ui->plot->axisRect()->setRangeZoomAxes(ui->plot->axisRect()->axes());
    ui->plot->axisRect()->setRangeDragAxes(ui->plot->axisRect()->axes());
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
    selection = selections->at(ui->cbSelection->currentIndex());
    std::vector<Stimulation> stim(selection.elites.size());
    for ( size_t i = 0; i < stim.size(); i++ ) {
        stim[i] = selection.elites[i].wave;
    }
    profiler.setStimulations(stim);

    std::vector<ErrorProfiler::Permutation> perm(lib.model.adjustableParams.size());
    perm[selection.param].n = 0; // Use full range for target param (see Permutation defaults)
    profiler.setPermutations(perm);

    ui->btnStart->setEnabled(false);
    ui->btnAbort->setEnabled(true);

    ui->log->addItem(QString("Profiling %1...").arg(ui->cbSelection->currentText()));
    ui->log->addItem("");
    ui->log->scrollToBottom();

    ui->plot->clearGraphs();

    emit profile();
}

void ProfileDialog::on_btnAbort_clicked()
{
    profiler.abort();
}

void ProfileDialog::profileComplete(int index)
{
    QListWidgetItem *item = ui->log->item(ui->log->count()-1);
    item->setText(QString("%1/%2 ...").arg(index+1).arg(selection.elites.size()));

    std::vector<scalar> const& prof = *(std::next(profiler.profiles.begin(), index));
    if ( prof.empty() )
        return;

    QCPGraph *graph = ui->plot->addGraph();
    for ( size_t i = 0; i < prof.size(); i++ )
        graph->addData(profiler.getParameterValue(selection.param, i), prof.at(i));
    graph->rescaleAxes(true);
    ui->plot->replot();
}

void ProfileDialog::done()
{
    ui->btnStart->setEnabled(true);
    ui->btnAbort->setEnabled(false);
    ui->log->addItem("Done.");
    ui->log->scrollToBottom();
}
