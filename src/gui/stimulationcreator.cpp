#include "stimulationcreator.h"
#include "ui_stimulationcreator.h"
#include "colorbutton.h"
#include "stimulationgraph.h"
#include "clustering.h"

constexpr static int nColours = 8;
const static QString colours[nColours] = {
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf"
};

StimulationCreator::StimulationCreator(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StimulationCreator),
    session(session),
    loadingStims(false),
    updatingStim(false),
    simulator(session.project.universal().createSimulator(0, session, session.qSettings(), false))
{
    ui->setupUi(this);
    ui->splitter->setStretchFactor(0,0);
    ui->splitter->setStretchFactor(1,1);
    ui->steps->setColumnWidth(0, 50);

    stimCopy.baseV = session.qStimulationData().baseV;
    stimCopy.duration = session.qStimulationData().duration;
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
        *stim = Stimulation(this->session.wavegen().getRandomStim(), this->session.wavegenData().dt);
        setStimulation();
    });

    connect(ui->saveSet, &QPushButton::clicked, [=](){
        WavesetCreator &creator = this->session.wavesets();
        this->session.queue(creator.actorName(), creator.actionManual, "", new ManualWaveset(stims), false);
    });
    connect(ui->saveDeck, &QPushButton::clicked, [=](){
        WavesetCreator &creator = this->session.wavesets();
        this->session.queue(creator.actorName(), creator.actionManualDeck, "", new ManualWaveset(stims), false);
    });

    connect(ui->copy, &QPushButton::clicked, [=](){
        stimCopy = *stim;
    });
    connect(ui->paste, &QPushButton::clicked, [=](){
        *stim = stimCopy;
        setStimulation();
    });

    connect(ui->diagnose, &QPushButton::clicked, this, &StimulationCreator::diagnose);

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectLegend);
    ui->plot->legend->setSelectableParts(QCPLegend::spItems);
    connect(ui->plot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->plot->selectedAxes();
        if ( axes.isEmpty() )
           axes = ui->plot->axisRect()->axes();
        ui->plot->axisRect()->setRangeZoomAxes(axes);
        ui->plot->axisRect()->setRangeDragAxes(axes);
    });
    connect(ui->plot, &QCustomPlot::axisDoubleClick, this, [=](QCPAxis *axis, QCPAxis::SelectablePart, QMouseEvent*) {
        axis->rescale(true);
        ui->plot->replot();
    });
    ui->plot->axisRect()->setRangeZoomAxes(ui->plot->axisRect()->axes());
    ui->plot->axisRect()->setRangeDragAxes(ui->plot->axisRect()->axes());
    ui->plot->xAxis->setLabel("Time [ms]");
    ui->plot->yAxis->setLabel("Voltage [mV]");
    ui->plot->yAxis2->setLabel("Current [nA]");

    // *** Traces ***
    int n = session.project.model().adjustableParams.size();
    ui->params->setColumnCount(n);
    ui->traceTable->setColumnCount(n+1);
    QStringList labels;
    for ( int i = 0; i < n; i++ ) {
        const AdjustableParam &p = session.project.model().adjustableParams.at(i);
        labels << QString::fromStdString(p.name);
        QDoubleSpinBox *widget = new QDoubleSpinBox;
        widget->setRange(-999999,999999);
        widget->setDecimals(3);
        widget->setValue(p.initial);
        ui->params->setCellWidget(0, i, widget);
        connect(widget, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), this, [=](double){
            paramsSrc = SrcManual;
        });
    }
    paramsSrc = SrcBase;
    ui->params->setHorizontalHeaderLabels(labels);
    labels.push_front("Source");
    ui->traceTable->setHorizontalHeaderLabels(labels);
    ui->tab_traces->setEnabled(false);

    connect(ui->paramSource, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, [=](int idx) {
        if ( this->session.gaFitter().results().empty() )
            return;
        int ep = ui->paramEpoch->value();
        const GAFitter::Output &fit = this->session.gaFitter().results().at(idx);
        ui->paramEpoch->setMaximum(fit.epochs);
        ui->paramEpoch->setMinimum(this->session.daqData(fit.resultIndex).simulate == -1 ? -3 : -2);
        if ( ep == ui->paramEpoch->value() )
            emit ui->paramEpoch->valueChanged(ui->paramEpoch->value());
    });
    connect(ui->paramEpoch, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, [=](int val) {
        if ( this->session.gaFitter().results().empty() )
            return;
        if ( val == -3 ) {
            ui->params->setEnabled(false);
            paramsSrc = SrcRec;
            return;
        }
        ui->params->setEnabled(true);
        const GAFitter::Output &fit = this->session.gaFitter().results().at(ui->paramSource->value());
        const std::vector<scalar> *params;
        if ( val == -2 )
            params =& fit.targets;
        else if ( val == -1 || val >= int(fit.epochs) )
            params =& fit.finalParams;
        else
            params =& fit.params[val];
        for ( size_t i = 0; i < params->size(); i++ )
            qobject_cast<QDoubleSpinBox*>(ui->params->cellWidget(0, i))->setValue(params->at(i));
        paramsSrc = SrcFit;
    });
    connect(ui->paramReset, &QPushButton::clicked, this, [=](){
        for ( size_t i = 0; i < this->session.project.model().adjustableParams.size(); i++ )
            qobject_cast<QDoubleSpinBox*>(ui->params->cellWidget(0, i))->setValue(this->session.project.model().adjustableParams[i].initial);
        paramsSrc = SrcBase;
    });
    auto updateParamSources = [=](){
        ui->paramSource->setMaximum(this->session.gaFitter().results().size()-1);
    };
    connect(&session.gaFitter(), &GAFitter::done, this, updateParamSources);
    updateParamSources();
    emit ui->paramSource->valueChanged(0); // ensure paramSource minimum is correctly set up

    connect(ui->traceTable, &QTableWidget::itemChanged, this, &StimulationCreator::traceEdited);
}

StimulationCreator::~StimulationCreator()
{
    delete ui;
    session.project.universal().destroySimulator(simulator);
    simulator = nullptr;
}

void StimulationCreator::updateSources()
{
    ui->sources->clear();
    ui->sources->addItem("Copy from...");
    ui->sources->addItem("Import external...");
    for ( WaveSource &src : session.wavesets().sources() ) {
        ui->sources->addItem(src.prettyName(), QVariant::fromValue(src));
    }
    ui->sources->setCurrentIndex(0);
}

void StimulationCreator::copySource()
{
    if ( ui->sources->currentIndex() == 0 ) {
        return;
    } else if ( ui->sources->currentIndex() == 1 ) {
        QString dop = QFileDialog::getOpenFileName(this, "Select project...", "", "*.dop");
        if ( dop.isEmpty() )
            return;

        QApplication::setOverrideCursor(Qt::WaitCursor);
        setEnabled(false);
        repaint();
        Project proj(dop, true);
        setEnabled(true);
        QApplication::restoreOverrideCursor();

        QStringList sessions = QDir(proj.dir() + "/sessions/").entryList(QDir::Dirs | QDir::NoDotAndDotDot, QDir::Name | QDir::Reversed);
        if ( sessions.isEmpty() )
            return;
        bool ok;
        QString selectedSession = QInputDialog::getItem(this, "Select session...", "Session:", sessions, 0, false, &ok);
        if ( !ok || selectedSession.isEmpty() )
            return;

        QApplication::setOverrideCursor(Qt::WaitCursor);
        setEnabled(false);
        repaint();
        Session ses(proj, proj.dir() + "/sessions/" + selectedSession);
        setEnabled(true);
        QApplication::restoreOverrideCursor();

        QStringList sources;
        int i = 0;
        for ( WaveSource &src : ses.wavesets().sources() )
            sources << QString("%1: %2").arg(QString::number(i++), src.prettyName());
        if ( sources.isEmpty() )
            return;
        QString sel = QInputDialog::getItem(this, "Select source...", "Source:", sources, 0, false, &ok);
        if ( !ok || sel.isEmpty() )
            return;
        i = sel.split(':').first().toInt();
        stims = ses.wavesets().sources()[i].stimulations();
    } else {
        stims = ui->sources->currentData().value<WaveSource>().stimulations();
    }

    QApplication::setOverrideCursor(Qt::WaitCursor);
    setEnabled(false);
    repaint();

    loadingStims = true;
    ui->nStim->setValue(stims.size());
    loadingStims = false;
    setNStims(stims.size());

    setEnabled(true);
    QApplication::restoreOverrideCursor();
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
    double minV = session.qStimulationData().minVoltage, maxV = session.qStimulationData().maxVoltage;
    ui->nSteps->setMaximum(Stimulation::maxSteps);
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

        ui->tab_traces->setEnabled(true);
    } else {
        ui->editor->setEnabled(false);
        ui->tab_traces->setEnabled(false);
    }

    redraw();

    if ( rows.size() == 1 )
        setupTraces();
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
        voltage->setRange(session.qStimulationData().minVoltage, session.qStimulationData().maxVoltage);
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

void StimulationCreator::makeHidable(QCPGraph *g)
{
    QCPPlottableLegendItem *item = ui->plot->legend->itemWithPlottable(g);
    connect(item, &QCPPlottableLegendItem::selectionChanged, [=](bool on){
        if ( !on ) return;
        on = g->visible();
        g->setVisible(!on);
        item->setTextColor(on ? Qt::lightGray : Qt::black);
        item->setSelected(false);
        ui->plot->replot();
    });
}

void StimulationCreator::diagnose()
{
    if ( session.busy() ) {
        QMessageBox::warning(this, "Busy", "Other actions are currently in progress. Pause or cancel before diagnosing.");
        return;
    }
    double dt = session.qRunData().dt;
    int nGraphs = session.project.model().adjustableParams.size() + 1;
    std::vector<double> norm(nGraphs-1, 1);

    iStimulation iStim(*stim, dt);
    session.wavegen().diagnose(iStim, dt, session.qRunData().simCycles);

    redraw();
    makeHidable(ui->plot->graph());

    QVector<double> keys(iStim.duration);
    QVector<double> mean(iStim.duration, 0), sd(iStim.duration, 0);
    std::vector<QVector<double>> values(nGraphs, QVector<double>(iStim.duration));
    scalar *delta = session.wavegen().lib.diagDelta;

    for ( int t = 0; t < iStim.duration; t++ ) {
        // Time
        keys[t] = t * dt;

        // (Normalised) deviations and mean
        for ( int i = 0; i < nGraphs; i++, delta++ ) {
            double val = *delta;// Pointer magic! Equivalent to lib.diagDelta[t * nGraphs + i]
            if ( i > 0 ) {
                val /= norm[i-1];
                mean[t] += val;
            }
            values[i][t] = val;
        }
        mean[t] /= nGraphs-1;

        // standard deviation
        for ( int i = 1; i < nGraphs; i++ ) {
            double val = mean[t] - values[i][t];
            sd[t] += val*val;
        }
        sd[t] = sqrt(sd[t]/(nGraphs-1));
    }

    clustering();

    QCPGraph *g = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis2);
    g->setName("Reference current");
    g->setData(keys, values[0], true);
    g->setPen(QPen(Qt::black));
    makeHidable(g);

    g = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis2);
    g->setName("Mean deviation");
    g->setData(keys, mean, true);
    g->setPen(QPen(Qt::gray));
    makeHidable(g);

    g = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis2);
    g->setName("Deviation s.d.");
    g->setData(keys, sd, true);
    g->setPen(QPen("#8888aa"));
    g->setBrush(QBrush("#668888aa"));
    makeHidable(g);

    for ( int i = 1; i < nGraphs; i++ ) {
        const AdjustableParam &p = session.wavegen().lib.adjustableParams[i-1];
        g = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis2);
        g->setName(QString::fromStdString(p.name));
        g->setData(keys, values[i], true);
        g->setPen(QPen(QColor(colours[i % nColours])));
        makeHidable(g);
    }

    ui->plot->yAxis2->setVisible(true);
    ui->plot->legend->setVisible(true);
    ui->plot->yAxis2->rescale();

    ui->plot->replot();
}

void StimulationCreator::clustering()
{
    const GAFitterSettings &settings = session.qGaFitterSettings();
    double dt = session.qRunData().dt;
    int nParams = session.project.model().adjustableParams.size();
    iStimulation iStim(*stim, dt);
    std::vector<double> norm(nParams, 1);

    std::vector<std::vector<Section>> clusters = constructClusters(iStim, session.wavegen().lib.diagDelta, settings.cluster_blank_after_step/dt,
                                                                   nParams+1, norm, settings.cluster_fragment_dur/dt, settings.cluster_threshold,
                                                                   settings.cluster_min_dur/settings.cluster_fragment_dur);
    std::vector<Section> bookkeeping;
    std::cout << "\n*** " << clusters.size() << " natural clusters for stimulation " << (stim-stims.begin()) << std::endl;
    for ( const std::vector<Section> &cluster : clusters ) {
        printCluster(cluster, nParams, dt);
        Section tmp {0, 0, std::vector<double>(nParams, 0)};
        for ( const Section &sec : cluster )
            for ( int i = 0; i < nParams; i++ )
                tmp.deviations[i] += sec.deviations[i];
        bookkeeping.push_back(std::move(tmp));
    }

    auto sim = constructSimilarityTable(bookkeeping, nParams);
    for ( const auto &col : sim ) {
        for ( double s : col )
            std::cout << '\t' << s;
        std::cout << std::endl;
    }

    std::cout << "\n*** Observation window section:" << std::endl;
    // Shortcut to construct one section for the entire observation window:
    std::vector<Section> tObs;
    constructSections(session.wavegen().lib.diagDelta, iStim.tObsBegin, iStim.tObsEnd, nParams+1, norm, iStim.tObsEnd-iStim.tObsBegin+1, tObs);
    printCluster(tObs, nParams, dt);

    std::vector<Section> primitives = constructSectionPrimitives(iStim, session.wavegen().lib.diagDelta, settings.cluster_blank_after_step/dt,
                                                                 nParams+1, norm, settings.cluster_fragment_dur/dt);
    std::vector<Section> sympa = findSimilarCluster(primitives, nParams, settings.cluster_threshold, tObs[0]);
    if ( !sympa.empty() ) {
        std::cout << "\n*** Sympathetic cluster:\n";
        printCluster(sympa, nParams, dt);
    } else {
        std::cout << "\n*** No sympathetic sections found.\n";
    }
    std::cout << std::endl;
}



void StimulationCreator::setupTraces()
{
    ui->traceTable->clearContents();
    ui->traceTable->setRowCount(0);
    for ( size_t i = 0; i < traces.size(); i++ ) {
        if ( traces[i].stim == *stim ) {
            addTrace(traces[i], i);
        }
    }
}

void StimulationCreator::on_paramTrace_clicked()
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    int nParams = session.project.model().adjustableParams.size();
    DAQ *daq = simulator;
    Stimulation actualStim = *stim;

    Trace trace;
    trace.stim = *stim;
    trace.params.resize(nParams, 0);
    if ( paramsSrc == SrcRec ) {
        const GAFitter::Output &fit = session.gaFitter().results().at(qobject_cast<QSpinBox*>(ui->paramSource)->value());
        DAQFilter *rec = new DAQFilter(session, session.getSettings(fit.resultIndex));
        if ( session.getLog()->entry(fit.resultIndex).timestamp < QDateTime(QDate(2018,9,20)) ) {
            std::cout << "Warning: Channel association and scaling may not reflect what the algorithm saw." << std::endl;
            QString cfg = fit.VCRecord;
            cfg.replace(".atf", ".cfg");
            if ( QFileInfo(cfg).exists() )
                session.loadConfig(cfg);
            rec->getCannedDAQ()->assoc = session.cdaq_assoc;
        } else {
            rec->getCannedDAQ()->assoc = fit.assoc;
        }
        std::vector<Stimulation> sanitisedStims = session.gaFitter().sanitiseDeck(stims);
        rec->getCannedDAQ()->setRecord(sanitisedStims, fit.VCRecord);
        actualStim = sanitisedStims[stim - stims.begin()];
        daq = rec;
    } else {
        for ( int i = 0; i < nParams; i++ ) {
            double p = qobject_cast<QDoubleSpinBox*>(ui->params->cellWidget(0, i))->value();
            trace.params[i] = p;
            simulator->setAdjustableParam(i, p);
        }
    }
    trace.dt = daq->rund.dt;

    if ( paramsSrc == SrcBase )
        trace.label = "Base model";
    else if ( paramsSrc == SrcManual )
        trace.label = "Manual";
    else if ( paramsSrc == SrcRec )
        trace.label = QString("Fit %1, recording").arg(ui->paramSource->value());
    else {
        QString epStr;
        int ep = ui->paramEpoch->value(), src = ui->paramSource->value();
        if ( ep == -2 )
            epStr = "target";
        else if ( ep == -1 || ep >= int(session.gaFitter().results().at(src).epochs) )
            epStr = "final";
        else
            epStr = QString("epoch %1").arg(ep);
        trace.label = QString("Fit %1, %2").arg(QString::number(src), epStr);
    }

    daq->run(actualStim, session.runData().settleDuration);
    for ( size_t iT = 0, iTEnd = session.runData().settleDuration/session.runData().dt; iT < iTEnd; iT++ )
        daq->next();
    trace.current.reserve(daq->samplesRemaining);
    trace.voltage.reserve(daq->samplesRemaining);
    while ( daq->samplesRemaining ) {
        daq->next();
        trace.current.push_back(daq->current);
        trace.voltage.push_back(daq->voltage);
    }

    traces.push_back(std::move(trace));
    addTrace(traces.back(), traces.size()-1);

    if ( paramsSrc == SrcRec )
        delete daq;

    QApplication::restoreOverrideCursor();
}

void StimulationCreator::addTrace(Trace &trace, int idx)
{
    QVector<double> keys(trace.current.size());
    for ( int i = 0; i < keys.size(); i++ )
        keys[i] = i * trace.dt;

    trace.gI = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis2);
    trace.gI->setData(keys, trace.current, true);
    trace.gI->setPen(QPen(QColorDialog::standardColor(idx%20)));
    makeHidable(trace.gI);

    trace.gV = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis);
    trace.gV->setData(keys, trace.voltage, true);
    trace.gV->setPen(QPen(QColorDialog::standardColor(idx%20 + 21)));
    makeHidable(trace.gV);

    ui->plot->yAxis2->setVisible(true);
    ui->plot->legend->setVisible(true);
    ui->plot->yAxis2->rescale();
    // Replot deferred to traceEdited()

    int nRows = ui->traceTable->rowCount();
    ui->traceTable->setRowCount(nRows + 1);
    QTableWidgetItem *label = new QTableWidgetItem(trace.label);
    label->setData(Qt::UserRole, idx);
    ui->traceTable->setItem(nRows, 0, label);
    for ( size_t i = 0; i < session.project.model().adjustableParams.size(); i++ ) {
        QTableWidgetItem *item = new QTableWidgetItem(QString::number(trace.params[i]));
        item->setFlags(Qt::ItemIsEnabled);
        ui->traceTable->setItem(nRows, i+1, item);
    }
}

void StimulationCreator::traceEdited(QTableWidgetItem *item)
{
    if ( item->column() == 0 ) {
        Trace &trace = traces[item->data(Qt::UserRole).toInt()];
        trace.label = item->text();
        trace.gI->setName(QString("%1, current").arg(trace.label));
        trace.gV->setName(QString("%1, voltage").arg(trace.label));
        ui->plot->replot();
    }
}



void StimulationCreator::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->plot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), file);
}
