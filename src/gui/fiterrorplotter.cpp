#include "fiterrorplotter.h"
#include "ui_fiterrorplotter.h"
#include <QFileDialog>
#include "clustering.h"
#include "populationsaver.h"
#include "supportcode.h"

constexpr static int nColours = 8;
static QString colours[nColours] = {
    "#e41a1c",
    "#377eb8",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf"
};

FitErrorPlotter::FitErrorPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::FitErrorPlotter)
{
    ui->setupUi(this);
    connect(ui->register_browse, &QToolButton::clicked, this, [=](){
        QString reg = QFileDialog::getOpenFileName(this, "Open register file...");
        if ( !reg.isEmpty() )
            ui->register_path->setText(reg);
    });
    connect(ui->params, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [=](int idx){
        ui->epoch->setEnabled(idx==2);
        replot();
    });

    connect(ui->protocols, &QTableWidget::itemSelectionChanged, this, [=](){
        std::vector<int> pidx = get_protocol_indices();
        if ( pidx.size() == 1 )
            ui->trace_stimidx->setMaximum(protocols[pidx[0]].stims.size()-1);
        replot();
    });

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_rec, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_sim, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_stim, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_stimidx, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &FitErrorPlotter::replot);
    connect(ui->trace_single, &QRadioButton::toggled, this, &FitErrorPlotter::replot);
    connect(ui->trace_single, &QRadioButton::toggled, ui->trace_stimidx, &QSpinBox::setEnabled);

    connect(ui->flipCats, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->groupCat, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &FitErrorPlotter::replot);
    connect(ui->splitTarget, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->protocolMeans, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->groupMeans, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
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

FitErrorPlotter::~FitErrorPlotter()
{
    delete lib;
    delete ui;
}

void FitErrorPlotter::init(Session *session)
{
    this->session = session;
}

void FitErrorPlotter::setData(std::vector<FitInspector::Group> data, bool summarising)
{
    this->data = data;
    this->summarising = summarising;
    quint32 maxEpoch = 0;

    // Count: n = number of fits; nRec = number of fits against recordings;
    // nFound = number of fits against registered recordings; nProtocolHits[X] = number of fits with matching protocol X
    int n = 0, nRec = 0, nFound = 0;
    std::vector<int> nProtocolHits(protocols.size(), 0);
    for ( const FitInspector::Group &g : data ) {
        for ( const FitInspector::Fit &f : g.fits ) {
            ++n;
            if ( session->daqData(f.fit().resultIndex).simulate == -1 ) {
                ++nRec;
                maxEpoch = std::max(maxEpoch, f.fit().epochs);
                QString cell = "";
                for ( const auto &reg : register_map ) {
                    if ( f.fit().VCRecord.endsWith(reg.second.file) ) {
                        cell = reg.second.cell;
                        ++nFound;
                        break;
                    }
                }
                if ( !cell.isEmpty() )
                    for ( size_t pi = 0; pi < protocols.size(); pi++ )
                        if ( register_map.find(std::make_pair(cell, protocols[pi].name)) != register_map.end() )
                            ++nProtocolHits[pi];
            }
        }
    }

    ui->matchedLabel->setText(QString("Fits matched against register: %1/%2").arg(nFound).arg(n));
    for ( size_t i = 0; i < protocols.size(); i++ )
        ui->protocols->setItem(i, 0, new QTableWidgetItem(QString("%1/%2").arg(nProtocolHits[i]).arg(nFound)));
    ui->epoch->setMaximum(maxEpoch);

    if ( isVisible() )
        replot();
}



void FitErrorPlotter::on_register_path_textChanged(const QString &arg1)
{
    QFile file(arg1);
    if ( !file.exists() || !file.open(QIODevice::ReadOnly | QIODevice::Text) ) {
        std::cout << "Register \"" << arg1 << "\" not found or unable to open." << std::endl;
        return;
    }
    register_map.clear();
    protocols.clear();
    register_dir = QDir(arg1);
    register_dir.cdUp();

    do {
        // Read register entry
        QString line = file.readLine().trimmed();
        QStringList fields = line.split('\t');
        if ( line.startsWith('#') || fields.size() < 10 )
            continue;

        RegisterEntry reg;
        reg.file = fields[0];
        reg.cell = fields[1];
        reg.protocol = fields[2];
        reg.rund.clampGain = fields[3].toDouble();
        reg.rund.accessResistance = fields[4].toDouble();
        reg.rund.Imax = fields[5].toDouble();
        reg.rund.settleDuration = fields[6].toDouble();

        reg.assoc.Iidx = fields[7].toInt();
        reg.assoc.Vidx = fields[8].toInt();
        reg.assoc.Iscale = fields[9].toDouble();

        // Read/load protocol
        auto pit = std::find_if(protocols.begin(), protocols.end(), [=](const Protocol &p){ return p.name == reg.protocol; });
        if ( pit == protocols.end() ) {
            Protocol protocol;
            protocol.name = reg.protocol;

            std::string path = register_dir.filePath(reg.protocol).toStdString();
            std::ifstream pfile(path);
            if ( !pfile.is_open() ) {
                std::cerr << "Failed to open protocol " << reg.protocol << " (" << path << ")." << std::endl;
                continue;
            }
            WavesetCreator::readStims(protocol.stims, pfile, protocol.dt);

            // Prepare iStims and observations
            protocol.istims.reserve(protocol.stims.size());
            protocol.iObs.reserve(protocol.stims.size());
            protocol.blankCycles = session->qWavegenData().cluster.blank/protocol.dt;
            for ( const Stimulation &stim : protocol.stims ) {
                protocol.istims.push_back(iStimulation(stim, protocol.dt));
                protocol.iObs.push_back(iObserveNoSteps(protocol.istims.back(), protocol.blankCycles));
            }

            protocol.idx = protocols.size();
            protocols.push_back(std::move(protocol));
            pit = protocols.end() - 1;
        }

        reg.rund.dt = pit->dt;

        reg.pprotocol =& *pit; // Temporary, for loadRecording() only
        if ( !loadRecording(reg, false) )
            continue;

        register_map[std::make_pair(reg.cell, reg.protocol)] = reg;
    } while ( !file.atEnd() );

    // Register definitive protocol pointers (at the end, to avoid ::protocols reallocating)
    for ( auto &regit : register_map )
        regit.second.pprotocol = &*std::find_if(protocols.begin(), protocols.end(), [=](const Protocol &p){
            return p.name == regit.second.protocol;
        });

    // Build protocol table
    {
        const QSignalBlocker blocker(ui->protocols);
        ui->protocols->clearContents();
        ui->protocols->setRowCount(protocols.size());
        for ( size_t i = 0; i < protocols.size(); i++ ) {
            ui->protocols->setItem(i, 1, new QTableWidgetItem(protocols[i].name));
        }
    }

    setData(data, summarising);
}

bool FitErrorPlotter::loadRecording(RegisterEntry &reg, bool readData)
{
    // Already loaded: return
    if ( !reg.data.empty() && reg.rund.VC == session->qRunData().VC )
        return true;

    // Read recording
    const std::vector<Stimulation> &stims = reg.pprotocol->stims;
    Settings settings = session->qSettings();
    settings.daqd.filter.active = false;
    settings.rund = reg.rund;
    CannedDAQ recording(*session, settings);
    recording.assoc = reg.assoc;
    if ( !recording.setRecord(stims, register_dir.filePath(reg.file), readData) )
        return false;

    if ( !readData )
        return true;

    // Write recording to register entry
    reg.data.resize(stims.size());
    for ( size_t stimIdx = 0; stimIdx < stims.size(); stimIdx++ ) {
        recording.run(stims[stimIdx]);
        reg.data[stimIdx].resize(reg.pprotocol->istims[stimIdx].duration);

//        size_t nextObs = 0;
        int iSample = 0;
        while ( iSample < reg.pprotocol->istims[stimIdx].duration /*&& nextObs < iObservations::maxObs*/) {
//            while ( iSample < reg.pprotocol->iObs[stimIdx].start[nextObs] ) {
//                recording.next();
//                ++iSample;
//            }
//            while ( iSample < reg.pprotocol->iObs[stimIdx].stop[nextObs] ) {
                recording.next();
                reg.data[stimIdx][iSample] = reg.rund.VC ? recording.current : recording.voltage;
                ++iSample;
//            }
//            ++nextObs;
        }
    }
    return true;
}

std::vector<RecStruct> FitErrorPlotter::get_requested_recordings(int &nTraces, int &maxStimLen)
{
    std::vector<RecStruct> recordings;
    nTraces = maxStimLen = 0;

    std::vector<int> protocol_indices = get_protocol_indices();
    if ( protocol_indices.empty() )
        return recordings;

    // Adjust observations to potentially changed settings, and find longest stim duration
    for ( int iProtocol : protocol_indices ) {
        Protocol &prot = protocols[iProtocol];
        int blankCycles = session->qWavegenData().cluster.blank/prot.dt;
        if ( blankCycles != prot.blankCycles ) {
            prot.blankCycles = blankCycles;
            for ( size_t i = 0; i < prot.stims.size(); i++ ) {
                prot.iObs[i] = iObserveNoSteps(prot.istims[i], blankCycles);
            }
        }
        for ( const iStimulation &I : prot.istims )
            maxStimLen = std::max(maxStimLen, I.duration);
    }

    // Find total number of target traces, and prepare list of fits & register entries
    for ( size_t iGroup = 0; iGroup < data.size(); iGroup++ ) {
        for ( size_t iFit = 0; iFit < data[iGroup].fits.size(); iFit++ ) {
            const FitInspector::Fit &f = data[iGroup].fits[iFit];

            // Discard fits that didn't target recordings
            if ( session->daqData(f.fit().resultIndex).simulate != -1 )
                continue;

            // Find the cell to which this fit was targeted
            QString cell = "";
            for ( const auto &reg : register_map ) {
                if ( f.fit().VCRecord.endsWith(reg.second.file) ) {
                    cell = reg.second.cell;
                    break;
                }
            }
            if ( cell.isEmpty() )
                continue;

            // Count traces and initialise recording numbers
            for ( int iProtocol : protocol_indices ) {
                auto reg_iter = register_map.find(std::make_pair(cell, protocols[iProtocol].name));
                if ( reg_iter == register_map.end() )
                    continue;
                auto reco_iter = std::find_if(recordings.begin(), recordings.end(), [=](const RecStruct &r){
                    return r.reg == &reg_iter->second;
                });
                if ( reco_iter == recordings.end() ) {
                    recordings.push_back(RecStruct{{}, &reg_iter->second});
                    nTraces += protocols[iProtocol].stims.size();
                    reco_iter = recordings.end() - 1;
                }

                reco_iter->fit_coords.push_back(std::make_pair(iGroup, iFit));
            }
        }
    }

    return recordings;
}

void FitErrorPlotter::setup_lib_for_validation(int nTraces, int maxStimLen, bool get_traces, bool average_summary)
{
    // Prepare lib
    if ( lib == nullptr )
        lib = new UniversalLibrary(session->project, false);
    lib->setSingularRund(false);
    lib->setSingularStim(false);
    lib->setSingularTarget(false);
    lib->resizeTarget(nTraces, maxStimLen);
    lib->resizeOutput(maxStimLen);

    lib->simCycles = session->qRunData().simCycles;
    lib->integrator = session->qRunData().integrator;
    lib->assignment = lib->assignment_base
            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED;
    if ( average_summary )
        lib->assignment |= ASSIGNMENT_SUMMARY_AVERAGE;
    if ( !session->qRunData().VC )
        lib->assignment |= ASSIGNMENT_PATTERNCLAMP | ASSIGNMENT_PC_REPORT_PIN;
    if ( get_traces )
        lib->assignment |= ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_NONE | ASSIGNMENT_TIMESERIES_ZERO_UNTOUCHED_SAMPLES;
    lib->summaryOffset = 0;
}

void FitErrorPlotter::on_run_clicked()
{
    int nTraces = 0, maxStimLen = 0;
    std::vector<RecStruct> recordings = get_requested_recordings(nTraces, maxStimLen);
    if ( recordings.empty() )
        return;

    QApplication::setOverrideCursor(Qt::WaitCursor);

    bool get_traces = ui->tabWidget->currentWidget() == ui->trace_tab;
    int parameterSourceSelection = get_parameter_selection();

    setup_lib_for_validation(nTraces, maxStimLen, get_traces, true);

    // Load models into lib
    size_t modelIdx = 0;
    size_t targetOffset = 0;
    std::vector<ResultKey> keys(lib->NMODELS);
    for ( RecStruct &rec : recordings) {
        RegisterEntry &reg = *rec.reg;
        loadRecording(reg);

        for ( size_t stimIdx = 0; stimIdx < reg.pprotocol->stims.size(); stimIdx++ ) {
            // Write target traces to lib
            for ( int iSample = 0; iSample < reg.data[stimIdx].size(); iSample++ )
                lib->target[targetOffset + stimIdx + iSample*lib->targetStride] = reg.data[stimIdx][iSample];

            // Write model parameters and settings
            for ( const std::pair<size_t,size_t> fit_coords : rec.fit_coords ) {

                // Check if data already exists or is already queued
                keys[modelIdx] = ResultKey(
                            reg.pprotocol->idx,
                            stimIdx,
                            data[fit_coords.first].fits[fit_coords.second].idx,
                            parameterSourceSelection);
                if ( (get_traces && traces.find(keys[modelIdx]) != traces.end())
                     || (!get_traces && summaries.find(keys[modelIdx]) != summaries.end())
                     || (std::find(keys.begin(), keys.begin()+modelIdx, keys[modelIdx]) != keys.begin()+modelIdx) )
                    continue;

                // Write parameter values
                const GAFitter::Output &fit = data[fit_coords.first].fits[fit_coords.second].fit();
                std::vector<scalar> paramValues;
                if ( parameterSourceSelection == -2 ) // "Target"
                    paramValues = fit.targets;
                else if ( parameterSourceSelection >= 0 && parameterSourceSelection < int(fit.epochs) ) // "Epoch..."
                        paramValues = fit.params.at(ui->epoch->value());
                else if ( fit.final )
                    paramValues = fit.finalParams;
                else
                    paramValues = fit.params.back();
                for ( size_t i = 0; i < paramValues.size(); i++ )
                    lib->adjustableParams[i][modelIdx] = paramValues[i];

                // Write settings
                lib->setRundata(modelIdx, reg.rund);
                lib->stim[modelIdx] = reg.pprotocol->istims[stimIdx];
                lib->obs[modelIdx] = reg.pprotocol->iObs[stimIdx];
                lib->targetOffset[modelIdx] = targetOffset + stimIdx;

                // Increment
                ++modelIdx;

                // Run a batch as soon as the model bucket is full
                if ( modelIdx == lib->NMODELS ) {
                    push_run_pull(keys, modelIdx, get_traces);
                    modelIdx = 0;
                }
            }
        }
        targetOffset += reg.pprotocol->stims.size();
    }

    // push-run-pull the remaining models
    if ( modelIdx > 0 ) {
        for ( size_t i = modelIdx; i < lib->NMODELS; i++ ) {
            lib->stim[i].duration = 0;
            lib->iSettleDuration[i] = 0;
        }
        push_run_pull(keys, modelIdx, get_traces);
    }

    replot();

    QApplication::restoreOverrideCursor();
}

void FitErrorPlotter::push_run_pull(std::vector<ResultKey> keys, size_t keySz, bool get_traces)
{
    lib->pushTarget();
    lib->push();
    lib->run();
    lib->pullSummary();
    if ( get_traces )
        lib->pullOutput();

    for ( size_t k = 0; k < keySz; k++ ) {
        summaries[keys[k]] = lib->summary[k];
        if ( get_traces ) {
            QVector<double> &trace = traces[keys[k]];
            trace.resize(lib->stim[k].duration);
            for ( int i = 0; i < trace.size(); i++ )
                trace[i] = lib->output[i*lib->NMODELS + k];
        }
    }
}



void FitErrorPlotter::replot()
{
    std::vector<int> protocol_indices = get_protocol_indices();
    bool traces_possible = protocol_indices.size() == 1 && data.size() == 1 && data[0].fits.size() == 1;
    ui->trace_tab->setEnabled(traces_possible);

    ui->plot->clearPlottables();

    if ( !protocol_indices.empty() && !data.empty() ) {
        if ( ui->tabWidget->currentWidget() == ui->trace_tab ) {
            if ( traces_possible )
                plot_traces(protocols[protocol_indices[0]]);
        } else {
            plot_boxes(protocol_indices);
        }
    }
    ui->plot->replot();
}

void FitErrorPlotter::plot_traces(Protocol &prot)
{
    ui->plot->yAxis2->setVisible(true);
    ui->plot->yAxis->setLabel("Current (nA)");
    ui->plot->yAxis2->setLabel("Voltage (mV)");
    ui->plot->xAxis->setLabel("Time (ms)");
    ui->plot->xAxis->setTicker(QSharedPointer<QCPAxisTicker>(new QCPAxisTicker));
    ui->plot->xAxis->setSubTicks(true);
    ui->plot->xAxis->setTickLength(5, 0);
    ui->plot->xAxis->grid()->setVisible(true);
    ui->plot->legend->setVisible(false);

    std::vector<int> stim_indices;
    if ( ui->trace_single->isChecked() )
        stim_indices.push_back(ui->trace_stimidx->value());
    else
        for ( size_t i = 0; i < prot.stims.size(); i++ )
            stim_indices.push_back(i);
    FitInspector::Fit &f = data[0].fits[0];

    QString cell = "";
    for ( const auto &reg : register_map ) {
        if ( f.fit().VCRecord.endsWith(reg.second.file) ) {
            cell = reg.second.cell;
            break;
        }
    }
    if ( cell.isEmpty() )
        return;

    auto reg_iter = register_map.find(std::make_pair(cell, prot.name));
    if ( reg_iter == register_map.end() )
        return;
    RegisterEntry &reg = reg_iter->second;

    bool found = false;
    for ( auto res : traces ) {
        if ( std::get<0>(res.first) == prot.idx && std::get<2>(res.first) == f.idx && std::get<3>(res.first) == get_parameter_selection() ) {
            found = true;
            break;
        }
    }
    if ( !found )
        return;

    QCPGraph *g;
    QVector<double> keys;
    auto sizeKeys = [&](int stimIdx){
        if ( keys.size() != prot.istims[stimIdx].duration ) {
            int keySize = keys.size();
            keys.resize(prot.istims[stimIdx].duration);
            if ( keySize < prot.istims[stimIdx].duration )
                for ( int i = keySize; i < prot.istims[stimIdx].duration; i++ )
                    keys[i] = i*prot.dt;
        }
    };

    if ( ui->trace_stim->isChecked() ) {
        QVector<double> stim;
        for ( int stimIdx : stim_indices ) {
            sizeKeys(stimIdx);
            stim.resize(prot.istims[stimIdx].duration);
            for ( int i = 0; i < stim.size(); i++ )
                stim[i] = getCommandVoltage(prot.stims[stimIdx], i*prot.dt);
            g = ui->plot->addGraph(ui->plot->xAxis, reg.rund.VC ? ui->plot->yAxis2 : ui->plot->yAxis);
            g->setData(keys, stim);
            g->setPen(QPen(Qt::blue));
        }
    }

    if ( ui->trace_rec->isChecked() ) {
        for ( int stimIdx : stim_indices ) {
            sizeKeys(stimIdx);
            g = ui->plot->addGraph(ui->plot->xAxis, reg.rund.VC ? ui->plot->yAxis : ui->plot->yAxis2);
            g->setData(keys, reg.data[stimIdx], true);
            g->setPen(QPen(Qt::green));
        }
    }

    if ( ui->trace_sim->isChecked() ) {
        for ( int stimIdx : stim_indices ) {
            sizeKeys(stimIdx);
            auto res = traces.find(std::make_tuple(prot.idx, stimIdx, f.idx, get_parameter_selection()));
            if ( res == traces.end() )
                continue;

            g = ui->plot->addGraph();
            g->setData(keys, res->second);
            g->setPen(QPen(Qt::red));
        }
    }

    ui->plot->rescaleAxes();
}

void FitErrorPlotter::plot_boxes(std::vector<int> protocol_indices)
{
    using std::swap;

    ui->plot->yAxis2->setVisible(false);
    ui->plot->yAxis->setLabel("RMS current error (nA)");
    ui->plot->xAxis->setLabel("");
    ui->plot->xAxis->setSubTicks(false);
    ui->plot->xAxis->setTickLength(0, 4);
    ui->plot->xAxis->grid()->setVisible(false);
    ui->plot->legend->setVisible(true);

    bool targetvsfinal = ui->splitTarget->isChecked();

    // Set up lists of global fit indices (one for each major group [cell or group/fit])
    QStringList fit_labels;
    std::vector<std::vector<int>> fits;
    std::vector<QColor> fit_colours;
    if ( ui->groupCat->currentIndex() == 0 && summarising ) {
        fits.reserve(data.size());
        for ( const FitInspector::Group &g : data ) {
            fits.emplace_back();
            fits.back().reserve(g.fits.size());
            for ( const FitInspector::Fit &f : g.fits )
                fits.back().push_back(f.idx);
            fit_labels.push_back(g.label);
            fit_colours.push_back(g.color);
        }
    } else if ( ui->groupCat->currentIndex() == 0 && !summarising ) {
        fits.reserve(data[0].fits.size());
        for ( const FitInspector::Fit &f : data[0].fits ) {
            fits.push_back({f.idx});
            fit_labels.push_back(f.label);
            fit_colours.push_back(f.color);
        }
    } else if ( ui->groupCat->currentIndex() == 1 ) {
        QStringList cells;
        std::vector<std::vector<int>> fits_sparse;
        for ( auto &reg_iter : register_map ) {
            int cellIdx;
            QString cell = reg_iter.first.first;
            if ( !cells.contains(cell) ) {
                cells.push_back(cell);
                fits_sparse.emplace_back();
                cellIdx = cells.size()-1;
            } else {
                cellIdx = cells.indexOf(QRegExp::escape(cell));
            }

            for ( const FitInspector::Group &g : data )
                for ( const FitInspector::Fit &f : g.fits )
                    if ( f.fit().VCRecord.endsWith(reg_iter.second.file) )
                        fits_sparse[cellIdx].push_back(f.idx);
        }
        for ( size_t i = 0; i < fits_sparse.size(); i++ ) {
            if ( !fits_sparse[i].empty() ) {
                fits.push_back(std::move(fits_sparse[i]));
                fit_labels.push_back(cells[i]);
                fit_colours.push_back(colours[i%nColours]);
            }
        }
    }

    // Collect all RMSE values at each [fits x protocol] intersection
    bool groupMeans = ui->groupMeans->isChecked();
    bool protMeans = ui->protocolMeans->isChecked();
    int source_selection = get_parameter_selection();
    int nFits = targetvsfinal ? fits.size()*2 : fits.size();
    if ( groupMeans )
        nFits += targetvsfinal ? 3 : 1;
    int nProtocols = protocol_indices.size() + int(protMeans);
    std::vector<std::vector<std::vector<double>>> rmse(nFits, std::vector<std::vector<double>>(nProtocols));
    for ( size_t i = 0; i < fits.size(); i++ ) {
        for ( size_t j = 0; j < protocol_indices.size(); j++ ) {
            for ( int k = 0; k < (targetvsfinal ? 2 : 1); k++ ) {
                std::vector<double> errors;
                for ( int fitIdx : fits[i] ) {
                    for ( size_t stimIdx = 0; stimIdx < protocols[protocol_indices[j]].stims.size(); stimIdx++ ) {
                        ResultKey key(protocol_indices[j], stimIdx, fitIdx, targetvsfinal ? k-2 : source_selection);
                        auto res_iter = summaries.find(key);
                        if ( res_iter != summaries.end() )
                            errors.push_back(res_iter->second);
                    }
                }
                rmse[targetvsfinal ? 2*i+k : i][j] = Quantile(errors, {0, 0.25, 0.5, 0.75, 1});

                // Accumulate errors across groups/protocols for means
                if ( groupMeans ) {
                    std::vector<double> *vec;
                    if ( targetvsfinal )
                        vec =& rmse[nFits - 3 + k][j];
                    else
                        vec =& rmse[nFits - 1][j];
                    vec->insert(vec->end(), errors.begin(), errors.end());
                }
                if ( protMeans ) {
                    std::vector<double> *vec =& rmse[targetvsfinal ? 2*i+k : i][nProtocols - 1];
                    vec->insert(vec->end(), errors.begin(), errors.end());
                }
            }
        }
    }

    // Consume the means in the margins
    if ( groupMeans ) {
        for ( size_t j = 0; j < protocol_indices.size(); j++ ) {
            if ( targetvsfinal ) {
                if ( protMeans ) {
                    rmse[nFits-3][nProtocols-1].insert(rmse[nFits-3][nProtocols-1].end(), rmse[nFits-3][j].begin(), rmse[nFits-3][j].end());
                    rmse[nFits-2][nProtocols-1].insert(rmse[nFits-2][nProtocols-1].end(), rmse[nFits-2][j].begin(), rmse[nFits-2][j].end());
                }
                std::vector<double> tmp;
                swap(rmse[nFits-3][j], tmp); // Fill tmp with "target" errors
                rmse[nFits-3][j] = Quantile(tmp, {0, 0.25, 0.5, 0.75, 1});
                tmp.insert(tmp.end(), rmse[nFits-2][j].begin(), rmse[nFits-2][j].end()); // add "fit" errors to tmp
                swap(rmse[nFits-1][j], tmp); // Fill overall mean vec with tmp for processing in the general section below
                rmse[nFits-2][j] = Quantile(rmse[nFits-2][j], {0, 0.25, 0.5, 0.75, 1});
            }
            if ( protMeans )
                rmse[nFits-1][nProtocols-1].insert(rmse[nFits-1][nProtocols-1].end(), rmse[nFits-1][j].begin(), rmse[nFits-1][j].end());
            rmse[nFits-1][j] = Quantile(rmse[nFits-1][j], {0, 0.25, 0.5, 0.75, 1});
        }
    }
    if ( protMeans )
        for ( int i = 0; i < nFits; i++ )
            rmse[i][nProtocols-1] = Quantile(rmse[i][nProtocols-1], {0, 0.25, 0.5, 0.75, 1});

    // Deal with major/minor flip
    bool flip = ui->flipCats->isChecked();
    auto el = [&rmse, flip](int minorIdx, int majorIdx) -> std::vector<double> {
        return flip ? rmse[minorIdx][majorIdx] : rmse[majorIdx][minorIdx];
    };
    int nMajor = flip ? nProtocols : nFits;
    int nMinor = flip ? nFits : nProtocols;

    QStringList majorLabels, minorLabels;
    for ( int iPro : protocol_indices )
        minorLabels.push_back(protocols[iPro].name.split('.').front());
    if ( protMeans )
        minorLabels.push_back("all");

    if ( targetvsfinal ) {
        for ( QString label : fit_labels )
            majorLabels << label + " target" << label + " fit";
        if ( groupMeans )
            majorLabels << "all target" << "all fit";
    } else {
        swap(majorLabels, fit_labels);
    }
    if ( groupMeans )
        majorLabels << "all";

    if ( flip )
        swap(minorLabels, majorLabels);

    std::vector<QColor> minorCols(nMinor);
    if ( flip )
        for ( int i = 0; i < nMinor; i++ )
            minorCols[i] = (targetvsfinal && i%2) ? fit_colours[i/2].lighter() : fit_colours[targetvsfinal ? i/2 : i];
    else
        for ( int i = 0; i < nMinor; i++ )
            minorCols[i] = QColor(colours[i%nColours]);

    // Set up ticks
    int stride = nMinor + 1;
    int tickOffset = nMinor/2;
    double barOffset = nMinor%2 ? 0 : 0.5;
    QSharedPointer<QCPAxisTickerText> textTicker(new QCPAxisTickerText);
    for ( int i = 0; i < majorLabels.size(); i++ )
        textTicker->addTick(tickOffset + i*stride, majorLabels[i]);
    ui->plot->xAxis->setTicker(textTicker);

    // Plot it!
    for ( int i = 0; i < nMinor; i++ ) {
        QCPStatisticalBox *box = new QCPStatisticalBox(ui->plot->xAxis, ui->plot->yAxis);
        box->setName(minorLabels[i]);
        QPen whiskerPen(Qt::SolidLine);
        whiskerPen.setCapStyle(Qt::FlatCap);
        box->setWhiskerPen(whiskerPen);
        box->setPen(QPen(minorCols[i]));
        QColor brushCol = minorCols[i];
        brushCol.setAlphaF(0.3);
        box->setBrush(QBrush(brushCol));
        box->setWidth(0.8);

        for ( int j = 0; j < nMajor; j++ )
            if ( !el(i,j).empty() )
                box->addData(j*stride + i + barOffset, el(i,j)[0], el(i,j)[1], el(i,j)[2], el(i,j)[3], el(i,j)[4]);
    }

    ui->plot->rescaleAxes();
    ui->plot->xAxis->setRange(-1, nMajor*stride - 1);
}

std::vector<int> FitErrorPlotter::get_protocol_indices()
{
    QList<QTableWidgetSelectionRange> selection = ui->protocols->selectedRanges();
    std::vector<int> rows;
    for ( auto range : selection )
        for ( int i = range.topRow(); i <= range.bottomRow(); i++ )
            rows.push_back(i);
    return rows;
}

int FitErrorPlotter::get_parameter_selection()
{
    int parameterSourceSelection = ui->params->currentIndex() - 2;
    if ( parameterSourceSelection == 0 )
        parameterSourceSelection = ui->epoch->value();
    return parameterSourceSelection;
}

void FitErrorPlotter::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->plot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), file);
}

void FitErrorPlotter::on_index_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select index file", session->directory());
    if ( file.isEmpty() )
        return;

    std::ofstream os(file.toStdString(), std::ios_base::out | std::ios_base::trunc);
    os << "fileno\tcell\tgroup\trecord\tn_epochs\tn_pop\tn_subpops\tsubpop_split\n";

    if ( lib == nullptr )
        lib = new UniversalLibrary(session->project, false);

    for ( size_t iGroup = 0; iGroup < data.size(); iGroup++ ) {
        for ( size_t iFit = 0; iFit < data[iGroup].fits.size(); iFit++ ) {
            const FitInspector::Fit &f = data[iGroup].fits[iFit];
            const GAFitterSettings settings = session->gaFitterSettings(f.fit().resultIndex);
            const DAQData daqd = session->daqData(f.fit().resultIndex);
            QString cell = "";

            if ( daqd.simulate == 1 ) {
                cell = (daqd.simd.paramSet==0 ? "base":QString("r%1").arg(f.fit().resultIndex));
                os << f.fit().resultIndex << '\t' << cell << '\t' << data[iGroup].label << '\t' << "simulated" << '\t'
                   << f.fit().epochs << '\t' << lib->NMODELS << '\t' << settings.num_populations << '\t' << settings.useDE << '\n';
                QDir fdir(file);
                fdir.cdUp();
                std::ofstream pfile(fdir.filePath(cell).toStdString() + ".params");
                for ( const scalar &t : f.fit().targets )
                    pfile << t << '\n';
                continue;
            }

            // Find the cell to which this fit was targeted
            for ( const auto &reg : register_map ) {
                if ( f.fit().VCRecord.endsWith(reg.second.file) ) {
                    cell = reg.second.cell;
                    break;
                }
            }
            if ( cell.isEmpty() )
                continue;


            os << f.fit().resultIndex << '\t' << cell << '\t' << data[iGroup].label << '\t' << f.fit().VCRecord << '\t'
               << f.fit().epochs << '\t' << lib->NMODELS << '\t' << settings.num_populations << '\t' << settings.useDE << '\n';
        }
    }
}

/// Returns a (subpopulation, epoch, param) vector of parameter values for fit target (1,1,*; source==0), median (source==1) or lowest error (source==2)
std::vector<std::vector<std::vector<scalar>>> FitErrorPlotter::get_param_values(const GAFitter::Output &fit, int source)
{
    if ( source == 0 ) { // target
        return {{fit.targets}};
    }

    const GAFitterSettings settings = session->gaFitterSettings(fit.resultIndex);
    if ( source == 2 && settings.num_populations == 1 ) { // best, full population
        return {fit.params};
    }

    PopLoader loader(QFile(session->resultFilePath(fit.resultIndex)), *lib);
    int popsz = lib->NMODELS / (settings.num_populations);
    if ( settings.useDE )
        popsz /= 2; // use ancestor population/errors only
    std::vector<std::vector<std::vector<scalar>>> ret(settings.num_populations,
                                                      std::vector<std::vector<scalar>>(fit.epochs,
                                                                                       std::vector<scalar>(lib->adjustableParams.size())));
    for ( size_t epoch = 0; epoch < fit.epochs; epoch++ ) {
        loader.load(epoch, *lib);
        for ( int pop = 0; pop < settings.num_populations; pop++ ) {
            size_t lo_offset = pop*popsz;
            size_t mid_offset = lo_offset + popsz/2;
            size_t hi_offset = lo_offset + popsz/2;

            if ( source == 1 ) { // median
                for ( size_t paramIdx = 0; paramIdx < lib->adjustableParams.size(); paramIdx++ ) {
                    AdjustableParam &P = lib->adjustableParams.at(paramIdx);
                    scalar *mid = P.v + mid_offset;
                    std::nth_element(P.v + lo_offset, mid, P.v + hi_offset);
                    ret[pop][epoch][paramIdx] = (*mid + *std::max_element(P.v + lo_offset, mid))/2;
                }
            } else { // best
                scalar *best = std::min_element(lib->summary + lo_offset, lib->summary + hi_offset);
                size_t best_offset = best - lib->summary;
                for ( size_t paramIdx = 0; paramIdx < lib->adjustableParams.size(); paramIdx++ ) {
                    ret[pop][epoch][paramIdx] = lib->adjustableParams.at(paramIdx)[best_offset];
                }
            }
        }
    }

    return ret;
}

void FitErrorPlotter::on_validate_clicked()
{
    if ( ui->validate_protocol->currentIndex() == 0 ) {
        crossvalidate(ui->validateTarget->currentIndex());
    } else if ( ui->validate_protocol->currentIndex() == 1 ) {
        validate(ui->validateTarget->currentIndex());
    } else if ( ui->validate_protocol->currentIndex() == 2 ) {
        for ( int i = 0; i < 3; i++ ) {
            crossvalidate(i);
            validate(i);
        }
    }
}

void FitErrorPlotter::crossvalidate(int target, std::vector<std::vector<std::vector<std::vector<std::vector<scalar>>>>> models /* (iGroup,iFit,subpop,epochparam) */)
{
    int nTraces = 0, maxStimLen = 0;
    std::vector<RecStruct> recordings = get_requested_recordings(nTraces, maxStimLen);
    if ( recordings.empty() )
        return;

    QApplication::setOverrideCursor(Qt::WaitCursor);

    setup_lib_for_validation(nTraces, maxStimLen, false, false);

    // Load parameter values from populations
    std::cout << "Loading parameter values..." << std::endl;
    std::vector<std::vector<std::vector<std::vector<scalar>>>> paramValues; // (fitIdx, subpopulation, epoch, param)
    std::vector<std::vector<std::vector<double>>> accumulated_summary; // one per model: (fitIdx, subpopulation, epoch)
    int nFits = 0;
    for ( RecStruct &rec : recordings ) {
        for ( const std::pair<size_t,size_t> fit_coords : rec.fit_coords ) {
            const FitInspector::Fit &f = data[fit_coords.first].fits[fit_coords.second];
            if ( nFits <= f.idx ) {
                nFits = f.idx + 1;
                paramValues.resize(nFits);
                accumulated_summary.resize(nFits);
            }
            if ( paramValues[f.idx].empty() ) {
                if ( models.empty() )
                    paramValues[f.idx] = get_param_values(f.fit(), target);
                else
                    paramValues[f.idx] = models[fit_coords.first][fit_coords.second];
                accumulated_summary[f.idx].resize(paramValues[f.idx].size(), std::vector<double>(paramValues[f.idx][0].size(), 0));
            }
        }
    }
    std::vector<int> total_observed_duration(paramValues.size(), 0); // One per fit: (fitIdx)

    std::vector<std::vector<std::vector<bool>>> queue_check(protocols.size()); // (protocol, stim, fitIdx)
    int nTotalStims = 0;
    for ( int iProtocol : get_protocol_indices() ) {
        queue_check[iProtocol].resize(protocols[iProtocol].stims.size(), std::vector<bool>(nFits, false));
        nTotalStims += ui->protocols->item(iProtocol, 0)->text().split('/')[0].toInt() * protocols[iProtocol].stims.size();
    }

    nTotalStims *= paramValues.back().size() * paramValues.back().back().size();
    std::cout << "Estimated simulation count: " << nTotalStims << ", " << ((nTotalStims+lib->NMODELS-1)/lib->NMODELS) << " batches." << std::endl;

    // Load models into lib
    size_t modelIdx = 0;
    size_t targetOffset = 0;
    size_t batch = 0;
    std::vector<double*> psummary(lib->NMODELS);
    for ( RecStruct &rec : recordings ) {
        RegisterEntry &reg = *rec.reg;
        loadRecording(reg);

        for ( size_t stimIdx = 0; stimIdx < reg.pprotocol->stims.size(); stimIdx++ ) {
            // Write target traces to lib
            for ( int iSample = 0; iSample < reg.data[stimIdx].size(); iSample++ )
                lib->target[targetOffset + stimIdx + iSample*lib->targetStride] = reg.data[stimIdx][iSample];

            // Write model parameters and settings
            for ( const std::pair<size_t,size_t> fit_coords : rec.fit_coords ) {

                // Check if data is already queued
                const FitInspector::Fit &f = data[fit_coords.first].fits[fit_coords.second];
                if ( queue_check[reg.pprotocol->idx][stimIdx][f.idx] )
                    continue;
                queue_check[reg.pprotocol->idx][stimIdx][f.idx] = true;

                // Add this observation duration to the total
                total_observed_duration[f.idx] += reg.pprotocol->iObs[stimIdx].duration();

                // Write one model for each subpop and epoch
                for ( size_t pop = 0; pop < paramValues[f.idx].size(); pop++ ) {
                    for ( size_t epoch = 0; epoch < paramValues[f.idx][pop].size(); epoch++ ) {

                        // Write parameter values
                        for ( size_t i = 0; i < lib->adjustableParams.size(); i++ )
                            lib->adjustableParams[i][modelIdx] = paramValues[f.idx][pop][epoch][i];

                        // Write settings
                        lib->setRundata(modelIdx, reg.rund);
                        lib->stim[modelIdx] = reg.pprotocol->istims[stimIdx];
                        lib->obs[modelIdx] = reg.pprotocol->iObs[stimIdx];
                        lib->targetOffset[modelIdx] = targetOffset + stimIdx;

                        // Keep tabs on where this goes
                        psummary[modelIdx] =& accumulated_summary[f.idx][pop][epoch];

                        // Increment
                        ++modelIdx;

                        // Run a batch as soon as the model bucket is full
                        if ( modelIdx == lib->NMODELS ) {
                            std::cout << "Simulating batch " << ++batch << " (state: cell " << reg.cell << ", protocol " << reg.pprotocol->name << ", stim " << stimIdx
                                      << ", fit " << f.idx << ", subpop " << pop << ", epoch " << epoch << ")..." << std::endl;
                            lib->pushTarget();
                            lib->push();
                            lib->run();
                            lib->pullSummary();
                            for ( size_t i = 0; i < modelIdx; i++ )
                                *psummary[i] += lib->summary[i];
                            modelIdx = 0;
                        }
                    }
                }
            }
        }
        targetOffset += reg.pprotocol->stims.size();
    }

    // push-run-pull the remaining models
    if ( modelIdx > 0 ) {
        for ( size_t i = modelIdx; i < lib->NMODELS; i++ ) {
            lib->stim[i].duration = 0;
            lib->iSettleDuration[i] = 0;
        }
        std::cout << "Simulating the final batch..." << std::endl;
        lib->pushTarget();
        lib->push();
        lib->run();
        lib->pullSummary();
        for ( size_t i = 0; i < modelIdx; i++ )
            *psummary[i] += lib->summary[i];
    }

    // Write
    std::cout << "Writing to files..." << std::endl;
    std::string suffix;
    if ( target == 0 )      suffix = ".target_xvalidation";
    else if ( target == 1 ) suffix = ".median_xvalidation";
    else if ( target == 2 ) suffix = ".lowerr_xvalidation";
    else if ( target == 3 ) suffix = ".final_xvalidation";
    for ( size_t fitIdx = 0; fitIdx < accumulated_summary.size(); fitIdx++ ) {
        if ( accumulated_summary[fitIdx].empty() )
            continue;
        const GAFitter::Output &fit = session->gaFitter().results().at(fitIdx);
        std::ofstream of(session->resultFilePath(fit.resultIndex).toStdString() + suffix, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        for ( size_t pop = 0; pop < accumulated_summary[fitIdx].size(); pop++ ) {
            for ( double &d : accumulated_summary[fitIdx][pop] )
                d = std::sqrt(d / total_observed_duration[fitIdx]); // -> RMSE
            of.write(reinterpret_cast<char*>(accumulated_summary[fitIdx][pop].data()), accumulated_summary[fitIdx][pop].size() * sizeof(double));
        }
    }

    std::cout << "Validation complete." << std::endl;
    QApplication::restoreOverrideCursor();
}

void FitErrorPlotter::validate(int target)
{
    QApplication::setOverrideCursor(Qt::WaitCursor);
    std::cout << "Loading parameter values..." << std::endl;

    // Load parameter values from populations
    std::vector<RegisterEntry*> recordings;
    std::vector<std::vector<FitInspector::Fit>> fits; // (rec, *)
    std::vector<std::vector<std::vector<std::vector<scalar>>>> paramValues; // (fitIdx, subpopulation, epoch, param)
    std::vector<std::vector<std::vector<double>>> accumulated_summary; // one per model: (fitIdx, subpopulation, epoch)
    int nModels = 0;
    for ( size_t iGroup = 0; iGroup < data.size(); iGroup++ ) {
        for ( size_t iFit = 0; iFit < data[iGroup].fits.size(); iFit++ ) {
            const FitInspector::Fit &f = data[iGroup].fits[iFit];

            // Discard fits that didn't target recordings
            if ( session->daqData(f.fit().resultIndex).simulate != -1 )
                continue;

            // Check for already selected recording
            bool found = false;
            for ( size_t iRec = 0; iRec < recordings.size(); iRec++ ) {
                if ( f.fit().VCRecord.endsWith(recordings[iRec]->file) ) {
                    fits[iRec].push_back(f);
                    found = true;
                    break;
                }
            }

            // Add new recording
            if ( !found ) {
                for ( auto &reg : register_map ) {
                    if ( f.fit().VCRecord.endsWith(reg.second.file) ) {
                        recordings.push_back(&reg.second);
                        fits.push_back({f});
                        found = true;
                        break;
                    }
                }
            }

            // Load parameter values
            if ( found ) {
                if ( int(paramValues.size()) <= f.idx ) {
                    paramValues.resize(f.idx+1);
                    accumulated_summary.resize(f.idx+1);
                }
                paramValues[f.idx] = get_param_values(f.fit(), target);
                accumulated_summary[f.idx].resize(paramValues[f.idx].size(), std::vector<double>(paramValues[f.idx][0].size(), 0));
                nModels += f.fit().stims.size() * paramValues[f.idx].size() * paramValues[f.idx][0].size();
            }
        }
    }

    int nTraces = 0, maxStimLen = 0;
    for ( RegisterEntry *rec : recordings ) {
        nTraces += rec->pprotocol->istims.size();
        for ( const iStimulation &I: rec->pprotocol->istims )
            maxStimLen = std::max(maxStimLen, I.duration);
    }

    std::vector<int> total_observed_duration(paramValues.size(), 0); // One per fit: (fitIdx)

    setup_lib_for_validation(nTraces, maxStimLen, false, false);

    std::cout << "Simulation count: " << nModels << ", " << ((nModels+lib->NMODELS-1)/lib->NMODELS) << " batches." << std::endl;

    // Load models into lib
    size_t modelIdx = 0;
    size_t targetOffset = 0;
    size_t batch = 0;
    std::vector<double*> psummary(lib->NMODELS);
    for ( size_t iRec = 0; iRec < recordings.size(); iRec++ ) {
        RegisterEntry &reg = *recordings[iRec];
        loadRecording(reg);

        for ( size_t stimIdx = 0; stimIdx < reg.pprotocol->stims.size(); stimIdx++ ) {
            // Write target traces to lib
            for ( int iSample = 0; iSample < reg.data[stimIdx].size(); iSample++ )
                lib->target[targetOffset + stimIdx + iSample*lib->targetStride] = reg.data[stimIdx][iSample];

            for ( size_t iFit = 0; iFit < fits[iRec].size(); iFit++ ) {
                const FitInspector::Fit &f = fits[iRec][iFit];

                // Add this observation duration to the total
                total_observed_duration[f.idx] += f.fit().obs[stimIdx].duration();

                // Write one model for each subpop and epoch
                for ( size_t pop = 0; pop < paramValues[f.idx].size(); pop++ ) {
                    for ( size_t epoch = 0; epoch < paramValues[f.idx][pop].size(); epoch++ ) {

                        // Write parameter values
                        for ( size_t i = 0; i < lib->adjustableParams.size(); i++ )
                            lib->adjustableParams[i][modelIdx] = paramValues[f.idx][pop][epoch][i];

                        // Write settings
                        lib->setRundata(modelIdx, reg.rund);
                        lib->stim[modelIdx] = f.fit().stims[stimIdx];
                        lib->obs[modelIdx] = f.fit().obs[stimIdx];
                        lib->targetOffset[modelIdx] = targetOffset + stimIdx;

                        // Keep tabs on where this goes
                        psummary[modelIdx] =& accumulated_summary[f.idx][pop][epoch];

                        // Increment
                        ++modelIdx;

                        // Run a batch as soon as the model bucket is full
                        if ( modelIdx == lib->NMODELS ) {
                            std::cout << "Simulating batch " << ++batch << " (state: protocol " << reg.pprotocol->name << ", stim " << stimIdx
                                      << ", fit " << f.idx << ", subpop " << pop << ", epoch " << epoch << ")..." << std::endl;
                            lib->pushTarget();
                            lib->push();
                            lib->run();
                            lib->pullSummary();
                            for ( size_t i = 0; i < modelIdx; i++ )
                                *psummary[i] += lib->summary[i];
                            modelIdx = 0;
                        }
                    }
                }
            }
        }
        targetOffset += reg.pprotocol->stims.size();
    }

    // push-run-pull the remaining models
    if ( modelIdx > 0 ) {
        for ( size_t i = modelIdx; i < lib->NMODELS; i++ ) {
            lib->stim[i].duration = 0;
            lib->iSettleDuration[i] = 0;
        }
        std::cout << "Simulating the final batch..." << std::endl;
        lib->pushTarget();
        lib->push();
        lib->run();
        lib->pullSummary();
        for ( size_t i = 0; i < modelIdx; i++ )
            *psummary[i] += lib->summary[i];
    }

    // Write
    std::cout << "Writing to files..." << std::endl;
    std::string suffix;
    if ( target == 0 )      suffix = ".target_validation";
    else if ( target == 1 ) suffix = ".median_validation";
    else if ( target == 2 ) suffix = ".lowerr_validation";
    for ( size_t fitIdx = 0; fitIdx < accumulated_summary.size(); fitIdx++ ) {
        if ( accumulated_summary[fitIdx].empty() )
            continue;
        const GAFitter::Output &fit = session->gaFitter().results().at(fitIdx);
        std::ofstream of(session->resultFilePath(fit.resultIndex).toStdString() + suffix, std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
        for ( size_t pop = 0; pop < accumulated_summary[fitIdx].size(); pop++ ) {
            for ( double &d : accumulated_summary[fitIdx][pop] )
                d = std::sqrt(d / total_observed_duration[fitIdx]); // -> RMSE
            of.write(reinterpret_cast<char*>(accumulated_summary[fitIdx][pop].data()), accumulated_summary[fitIdx][pop].size() * sizeof(double));
        }
    }

    std::cout << "Validation complete." << std::endl;
    QApplication::restoreOverrideCursor();
}

// Returns (npops, 1, nparams) finalists with the lowest cost across all fit/obs, and populates cost with the associated (npops) RMSEs
std::vector<std::vector<std::vector<scalar>>> find_best_finalists(Session *session, UniversalLibrary *lib, const GAFitter::Output &f, std::vector<double> &cost)
{
    // Setup
    for ( size_t i = 0; i < lib->adjustableParams.size(); i++ ) {
        for ( size_t j = 0; j < lib->NMODELS; j++ ) {
            lib->adjustableParams[i][j] = f.resume.population[i][j];
        }
    }

    const RunData rd = session->runData(f.resultIndex);
    lib->setSingularRund(true);
    lib->simCycles = rd.simCycles;
    lib->integrator = rd.integrator;
    lib->setRundata(0, rd);

    lib->setSingularTarget(true);

    lib->setSingularStim(true);

    lib->summaryOffset = 0;
    lib->stim[0] = f.stims[0];
    lib->push();

    // settle
    lib->assignment = lib->assignment_base | ASSIGNMENT_SETTLE_ONLY;
    lib->run();
    lib->iSettleDuration[0] = 0;
    lib->push(lib->iSettleDuration);

    lib->assignment = lib->assignment_base
            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED;

    Settings cfg = session->getSettings(f.resultIndex);
    DAQFilter *daq = new DAQFilter(*session, cfg);
    if ( session->daqData(f.resultIndex).simulate < 0 ) {
        daq->getCannedDAQ()->assoc = f.assoc;
        daq->getCannedDAQ()->setRecord(f.stimSource.stimulations(), f.VCRecord);
    } else {
        for ( size_t i = 0; i < lib->adjustableParams.size(); i++ )
            daq->setAdjustableParam(i, f.targets[i]);
    }

    // Run all stims
    int tTotal = 0;
    for ( const iStimulation &I : f.stims ) {
        lib->stim[0] = I;
        lib->obs[0] = iObserveNoSteps(I, session->wavegenData(f.resultIndex).cluster.blank/rd.dt);
        tTotal += lib->obs[0].duration();
        lib->push(lib->stim);
        lib->push(lib->obs);

        lib->resizeTarget(1, I.duration);

        // Initiate DAQ stimulation
        daq->reset();
        daq->run(Stimulation(I, rd.dt), rd.settleDuration);
        for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
            daq->next();


        // Step DAQ through full stimulation
        for ( int iT = 0; iT < I.duration; iT++ ) {
            daq->next();
            lib->target[iT] = daq->current;
        }

        // Run lib against target
        lib->pushTarget();
        lib->run();

        lib->assignment |= ASSIGNMENT_SUMMARY_PERSIST;
    }

    // Evaluate
    lib->pullSummary();
    std::vector<GAFitter::errTupel> f_err(lib->NMODELS);
    for ( size_t i = 0; i < lib->NMODELS; i++ ) {
        f_err[i].idx = i;
        f_err[i].err = lib->summary[i];
    }

    const GAFitterSettings gafs = session->gaFitterSettings(f.resultIndex);
    std::vector<std::vector<std::vector<scalar>>> best_models(gafs.num_populations, std::vector<std::vector<scalar>>(1, std::vector<scalar>(lib->adjustableParams.size())));
    cost.resize(gafs.num_populations);

    if ( gafs.useDE ) {
        int nmodels = lib->NMODELS / gafs.num_populations / 2;
        for ( int i = 0; i < gafs.num_populations; i++ ) {
            auto parent = std::min_element(f_err.begin() + nmodels * i, f_err.begin() + nmodels * (i+1), &GAFitter::errTupelSort);
            auto child = std::min_element(f_err.begin() + nmodels * i + lib->NMODELS/2, f_err.begin() + nmodels * (i+1) + lib->NMODELS/2, &GAFitter::errTupelSort);
            int idx = parent->err < child->err ? parent->idx : child->idx;
            for ( size_t j = 0; j < lib->adjustableParams.size(); j++ )
                best_models[i][0][j] = lib->adjustableParams[j][idx];
            cost[i] = std::sqrt(lib->summary[idx] / tTotal);
        }
    } else {
        int nmodels = lib->NMODELS / gafs.num_populations;
        for ( int i = 0; i < gafs.num_populations; i++ ) {
            auto winner = std::min_element(f_err.begin() + nmodels*i, f_err.begin() + nmodels*(i+1), &GAFitter::errTupelSort);
            for ( size_t j = 0; j < lib->adjustableParams.size(); j++ )
                best_models[i][0][j] = lib->adjustableParams[j][winner->idx];
            cost[i] = std::sqrt(winner->err / tTotal);
        }
    }

    return best_models;
}

void FitErrorPlotter::on_finalise_clicked()
{
    if ( lib == nullptr )
        lib = new UniversalLibrary(session->project, false);
    std::vector<std::vector<std::vector<std::vector<std::vector<scalar>>>>> finalists(data.size());
    for ( size_t iGroup = 0; iGroup < data.size(); iGroup++ ) {
        finalists[iGroup].resize(data[iGroup].fits.size());
        for ( size_t iFit = 0; iFit < data[iGroup].fits.size(); iFit++ ) {
            const FitInspector::Fit &f = data[iGroup].fits[iFit];
            std::vector<double> cost;

            std::ofstream pof(session->resultFilePath(f.fit().resultIndex).toStdString() + ".final.pop", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
            for ( size_t i = 0; i < lib->adjustableParams.size(); i++ )
                pof.write(reinterpret_cast<const char*>(f.fit().resume.population[i].data()), lib->NMODELS * sizeof(scalar));

            std::cout << "Finalising resultIdx " << f.fit().resultIndex << ", fit " << iFit << " in group " << iGroup << std::endl;
            finalists[iGroup][iFit] = find_best_finalists(session, lib, f.fit(), cost);
            std::ofstream of(session->resultFilePath(f.fit().resultIndex).toStdString() + ".final.params", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
            for ( std::vector<std::vector<scalar>> b : finalists[iGroup][iFit] )
                of.write(reinterpret_cast<char*>(b[0].data()), b[0].size() * sizeof(scalar));
            std::ofstream cof(session->resultFilePath(f.fit().resultIndex).toStdString() + ".final.cost", std::ios_base::out | std::ios_base::binary | std::ios_base::trunc);
            cof.write(reinterpret_cast<char*>(cost.data()), cost.size() * sizeof(double));
        }
    }

    if ( session->daqData().simulate == -1 )
        crossvalidate(3, finalists);
}
