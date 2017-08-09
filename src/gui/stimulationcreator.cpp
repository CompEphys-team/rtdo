#include "stimulationcreator.h"
#include "ui_stimulationcreator.h"
#include "colorbutton.h"
#include "stimulationgraph.h"

StimulationCreator::StimulationCreator(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StimulationCreator),
    session(session),
    loadingStims(false),
    updatingStim(false)
{
    ui->setupUi(this);
    ui->splitter->setStretchFactor(0,0);
    ui->splitter->setStretchFactor(1,1);
    ui->steps->setColumnWidth(0, 50);

    stimCopy.baseV = session.stimulationData().baseV;
    stimCopy.duration = session.stimulationData().duration;
    stimCopy.tObsBegin = 100;
    stimCopy.tObsEnd = 105;
    stimCopy.clear();

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateSources()));
    connect(ui->sources, SIGNAL(activated(int)), this, SLOT(copySource()));
    updateSources();

    connect(&session, SIGNAL(actionLogged(QString,QString,QString,int)), this, SLOT(setLimits()));
    setLimits();

    ui->nStim->setValue(session.project.model().adjustableParams.size());
    connect(ui->nStim, SIGNAL(valueChanged(int)), this, SLOT(setNStims(int)));
    setNStims(ui->nStim->value());

    connect(ui->stimulations, SIGNAL(itemSelectionChanged()), this, SLOT(setStimulation()));
    connect(ui->nSteps, SIGNAL(valueChanged(int)), this, SLOT(setNSteps(int)));
    connect(ui->duration, SIGNAL(editingFinished()), this, SLOT(setLimits()));

    connect(ui->duration, SIGNAL(valueChanged(double)), this, SLOT(updateStimulation()));
    connect(ui->baseV, SIGNAL(valueChanged(double)), this, SLOT(updateStimulation()));
    connect(ui->tObsBegin, SIGNAL(valueChanged(double)), this, SLOT(updateStimulation()));
    connect(ui->tObsEnd, SIGNAL(valueChanged(double)), this, SLOT(updateStimulation()));
    connect(ui->steps, SIGNAL(cellChanged(int,int)), this, SLOT(updateStimulation()));

    connect(ui->randomise, &QPushButton::clicked, [=](){
        *stim = this->session.wavegen().getRandomStim();
        setStimulation();
    });

    connect(ui->saveSet, &QPushButton::clicked, [=](){
        this->session.wavesets().makeManual(stims);
    });
    connect(ui->saveDeck, &QPushButton::clicked, [=](){
        this->session.wavesets().makeManual(stims);
        size_t idx = this->session.wavesets().manuals().size()-1;
        std::vector<WaveSource> src;
        for ( size_t i = 0; i < stims.size(); i++ ) {
            src.emplace_back(this->session, WaveSource::Manual, idx, i);
        }
        this->session.wavesets().makeDeck(src);
    });

    connect(ui->copy, &QPushButton::clicked, [=](){
        stimCopy = *stim;
    });
    connect(ui->paste, &QPushButton::clicked, [=](){
        *stim = stimCopy;
        setStimulation();
    });
}

StimulationCreator::~StimulationCreator()
{
    delete ui;
}

void StimulationCreator::updateSources()
{
    ui->sources->clear();
    ui->sources->addItem("Copy from...");
    for ( WaveSource &src : session.wavesets().sources() ) {
        ui->sources->addItem(src.prettyName(), QVariant::fromValue(src));
    }
    ui->sources->setCurrentIndex(0);
}

void StimulationCreator::copySource()
{
    if ( ui->sources->currentIndex() == 0 )
        return;
    loadingStims = true;
    stims = ui->sources->currentData().value<WaveSource>().stimulations();
    ui->nStim->setValue(stims.size());
    loadingStims = false;
    setNStims(stims.size());
}

void StimulationCreator::setNStims(int n)
{
    if ( loadingStims )
        return;
    stims.resize(n, stimCopy);
    ui->stimulations->clearContents();
    ui->stimulations->setRowCount(stims.size());
    bool named = (n == session.project.model().adjustableParams.size());
    QStringList labels;
    for ( int row = 0; row < n; row++ ) {
        ColorButton *c = new ColorButton();
        c->setColor(QColorDialog::standardColor(row % 42));
        connect(c, SIGNAL(colorChanged(QColor)), this, SLOT(redraw()));
        ui->stimulations->setCellWidget(row, 0, c);
        ui->stimulations->setItem(row, 1, new QTableWidgetItem(QString("%1 ms, %2 steps").arg(stims[row].duration).arg(stims[row].size())));
        labels << (named ? QString::fromStdString(session.project.model().adjustableParams.at(row).name) : QString::number(row));
    }
    ui->stimulations->setVerticalHeaderLabels(labels);
    ui->saveDeck->setEnabled(named);
}

void StimulationCreator::setLimits()
{
    double minV = session.stimulationData().minVoltage, maxV = session.stimulationData().maxVoltage;
    ui->nSteps->setMaximum(Stimulation::maxSteps - 1);
    ui->baseV->setRange(minV, maxV);
    ui->tObsBegin->setMaximum(ui->duration->value());
    ui->tObsEnd->setMaximum(ui->duration->value());
    for ( int i = 0; i < ui->steps->rowCount(); i++ ) {
        qobject_cast<QDoubleSpinBox*>(ui->steps->cellWidget(i, 1))->setMaximum(ui->duration->value());
        qobject_cast<QDoubleSpinBox*>(ui->steps->cellWidget(i, 2))->setRange(minV, maxV);
    }
    ui->plot->yAxis->setRange(minV, maxV);
}

void StimulationCreator::setStimulation()
{
    QList<QTableWidgetSelectionRange> selection = ui->stimulations->selectedRanges();
    std::vector<int> rows;
    for ( auto range : selection )
        for ( int i = range.topRow(); i <= range.bottomRow(); i++)
            rows.push_back(i);
    if ( rows.size() == 1 ) {
        updatingStim = true;
        stim = stims.begin() + rows[0];
        ui->editor->setEnabled(true);
        ui->nSteps->setValue(stim->size());
        ui->duration->setValue(stim->duration);
        setLimits();
        ui->baseV->setValue(stim->baseV);
        ui->tObsBegin->setValue(stim->tObsBegin);
        ui->tObsEnd->setValue(stim->tObsEnd);
        updatingStim = false;
        setNSteps(stim->size());
    } else {
        ui->editor->setEnabled(false);
    }

    redraw();
}

void StimulationCreator::setNSteps(int n)
{
    bool isUpdating = updatingStim;
    updatingStim = true;
    ui->steps->clearContents();
    ui->steps->setRowCount(n);
    while ( (int)stim->size() < n )
        stim->insert(stim->end(), *stim->end());
    while ( n < (int)stim->size() )
        stim->erase(stim->end()-1);
    for ( int i = 0; i < n; i++ ) {
        QTableWidgetItem *ramp = new QTableWidgetItem();
        ramp->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
        ramp->setCheckState(stim->steps[i].ramp ? Qt::Checked : Qt::Unchecked);
        ui->steps->setItem(i, 0, ramp);

        QDoubleSpinBox *time = new QDoubleSpinBox();
        time->setRange(0, ui->duration->value());
        time->setValue(stim->steps[i].t);
        ui->steps->setCellWidget(i, 1, time);
        connect(time, SIGNAL(valueChanged(double)), this, SLOT(updateStimulation()));

        QDoubleSpinBox *voltage = new QDoubleSpinBox();
        voltage->setRange(session.stimulationData().minVoltage, session.stimulationData().maxVoltage);
        voltage->setValue(stim->steps[i].V);
        ui->steps->setCellWidget(i, 2, voltage);
        connect(voltage, SIGNAL(valueChanged(double)), this, SLOT(updateStimulation()));
    }
    if ( !isUpdating ) {
        updatingStim = false;
        updateStimulation();
    }
}

void StimulationCreator::updateStimulation()
{
    if ( updatingStim )
        return;
    stim->duration = ui->duration->value();
    stim->baseV = ui->baseV->value();
    stim->tObsBegin = ui->tObsBegin->value();
    stim->tObsEnd = ui->tObsEnd->value();
    for ( int i = 0; i < ui->steps->rowCount(); i++ ) {
        stim->steps[i].ramp = ui->steps->item(i, 0)->checkState() == Qt::Checked;
        stim->steps[i].t = qobject_cast<QDoubleSpinBox*>(ui->steps->cellWidget(i, 1))->value();
        stim->steps[i].V = qobject_cast<QDoubleSpinBox*>(ui->steps->cellWidget(i, 2))->value();
    }

    ui->stimulations->item(stim - stims.begin(), 1)->setText(QString("%1 ms, %2 steps").arg(stim->duration).arg(stim->size()));

    redraw();
}

void StimulationCreator::redraw()
{
    ui->plot->clearGraphs();
    QList<QTableWidgetSelectionRange> selection = ui->stimulations->selectedRanges();
    std::vector<int> rows;
    for ( auto range : selection )
        for ( int i = range.topRow(); i <= range.bottomRow(); i++)
            rows.push_back(i);

    double tMax = 0;
    for ( int row : rows ) {
        StimulationGraph *g = new StimulationGraph(ui->plot->xAxis, ui->plot->yAxis, stims[row]);
        QColor col = qobject_cast<ColorButton*>(ui->stimulations->cellWidget(row, 0))->color;
        g->setPen(QPen(col));
        col.setAlphaF(0.2);
        g->setBrush(QBrush(col));
        tMax = std::max(tMax, double(stims[row].duration));
    }
    ui->plot->xAxis->setRange(0, tMax);
    ui->plot->replot();
}
