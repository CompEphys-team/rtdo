#include "profiledialog.h"
#include "ui_profiledialog.h"
#include "project.h"

ProfileDialog::ProfileDialog(Session *s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProfileDialog),
    session(s),
    profiler(session->profiler()),
    selections(nullptr)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);

    connect(this, SIGNAL(profile()), &profiler, SLOT(profile()));
    connect(&profiler, SIGNAL(profileComplete(int)), this, SLOT(profileComplete(int)));
    connect(&profiler, SIGNAL(done()), this, SLOT(done()));

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
    ui->plot->xAxis->setLabel("Candidate model's target parameter value");
    ui->plot->yAxis->setLabel("Error");

    connect(ui->xRange, &QCheckBox::stateChanged, [this](int state){
        ui->sbMin->setEnabled(state == Qt::Checked);
        ui->sbMax->setEnabled(state == Qt::Checked);
        int i = ui->cbSelection->currentIndex();
        if ( state == Qt::Unchecked && i >= 0) {
            ui->sbMin->setValue(this->profiler.lib.model.adjustableParams[selections->at(i).param].min);
            ui->sbMax->setValue(this->profiler.lib.model.adjustableParams[selections->at(i).param].max);
        }
    });
    connect(ui->cbSelection, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [this](int i){
        auto range = std::make_pair(this->profiler.lib.model.adjustableParams[selections->at(i).param].min,
                                    this->profiler.lib.model.adjustableParams[selections->at(i).param].max);
        ui->sbMin->setRange(range.first, range.second);
        ui->sbMax->setRange(range.first, range.second);
        if ( !ui->xRange->isChecked() ) {
            ui->sbMin->setValue(range.first);
            ui->sbMax->setValue(range.second);
        }
    });
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

    std::vector<ErrorProfiler::Permutation> perm(profiler.lib.model.adjustableParams.size());
    perm[selection.param].n = 0;
    if ( ui->xRange->isChecked() ) {
        perm[selection.param].min = ui->sbMin->value();
        perm[selection.param].max = ui->sbMax->value();
    } else {
        perm[selection.param].min = profiler.lib.model.adjustableParams[selection.param].min;
        perm[selection.param].max = profiler.lib.model.adjustableParams[selection.param].max;
    }
    profiler.setPermutations(perm);

    ui->btnStart->setEnabled(false);
    ui->btnAbort->setEnabled(true);

    ui->log->addItem(QString("Profiling %1...").arg(ui->cbSelection->currentText()));
    ui->log->addItem("");
    ui->log->scrollToBottom();

    ui->plot->clearGraphs();
    ui->plot->xAxis->setRange(perm[selection.param].min, perm[selection.param].max);
    ui->plot->yAxis->setRange(0, 1000);

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
    ui->plot->replot();
}

void ProfileDialog::done()
{
    ui->btnStart->setEnabled(true);
    ui->btnAbort->setEnabled(false);
    ui->log->addItem("Done.");
    ui->log->scrollToBottom();
}

void ProfileDialog::on_btnReplot_clicked()
{
    ui->plot->clearGraphs();
    for ( std::vector<scalar> const& prof : profiler.profiles ) {
        if ( prof.empty() )
            continue;
        QCPGraph *graph = ui->plot->addGraph();
        for ( size_t i = 0; i < prof.size(); i++ )
            graph->addData(profiler.getParameterValue(selection.param, i), prof.at(i));
    }
    ui->plot->rescaleAxes();
    ui->plot->xAxis->setRange(profiler.permutations[selection.param].min, profiler.permutations[selection.param].max);
    ui->plot->yAxis->setRange(0, 1000);
    ui->plot->replot();
}
