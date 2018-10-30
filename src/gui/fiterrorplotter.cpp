#include "fiterrorplotter.h"
#include "ui_fiterrorplotter.h"
#include <QFileDialog>
#include "clustering.h"

#include "supportcode.h"

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
    });

    ui->protocols->setColumnWidth(0, 22);
    connect(ui->protocols, &QTableWidget::cellChanged, this, [=](int, int col){
        if ( col == 0 ) {
            std::vector<int> pidx = get_protocol_indices();
            if ( pidx.size() == 1 )
                ui->trace_stimidx->setMaximum(protocols[pidx[0]].stims.size()-1);
            replot();
        }
    });

    connect(ui->tabWidget, &QTabWidget::currentChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_rec, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_sim, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_stim, &QCheckBox::stateChanged, this, &FitErrorPlotter::replot);
    connect(ui->trace_stimidx, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &FitErrorPlotter::replot);
    connect(ui->trace_single, &QRadioButton::toggled, this, &FitErrorPlotter::replot);
    connect(ui->trace_single, &QRadioButton::toggled, ui->trace_stimidx, &QSpinBox::setEnabled);

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

    ui->plot->yAxis2->setVisible(true);
    ui->plot->yAxis->setLabel("Current (nA)");
    ui->plot->yAxis2->setLabel("Voltage (mV)");
    ui->plot->xAxis->setLabel("Time (ms)");
}

FitErrorPlotter::~FitErrorPlotter()
{
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
        ui->protocols->setItem(i, 1, new QTableWidgetItem(QString("%1/%2").arg(nProtocolHits[i]).arg(nFound)));
    ui->epoch->setMaximum(maxEpoch);

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
            protocol.blankCycles = session->qGaFitterSettings().cluster_blank_after_step/protocol.dt;
            for ( const Stimulation &stim : protocol.stims ) {
                protocol.istims.push_back(iStimulation(stim, protocol.dt));

                iObservations obs;
                std::vector<std::pair<int,int>> pairs = observeNoSteps(protocol.istims.back(), protocol.blankCycles);
                for ( size_t j = 0; j < pairs.size() && j < iObservations::maxObs; j++ ) {
                    obs.start[j] = pairs[j].first;
                    obs.stop[j] = pairs[j].second;
                }
                protocol.iObs.push_back(obs);
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
            QTableWidgetItem *cb = new QTableWidgetItem();
            cb->setFlags(Qt::ItemIsEnabled | Qt::ItemIsUserCheckable);
            cb->setCheckState(Qt::Unchecked);
            ui->protocols->setItem(i, 0, cb);

            ui->protocols->setItem(i, 2, new QTableWidgetItem(protocols[i].name));
        }
    }

    setData(data, summarising);
}

bool FitErrorPlotter::loadRecording(RegisterEntry &reg, bool readData)
{
    // Already loaded: return
    if ( !reg.data.empty() )
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

        size_t nextObs = 0;
        int iSample = 0;
        while ( iSample < reg.pprotocol->istims[stimIdx].duration && nextObs < iObservations::maxObs) {
            while ( iSample < reg.pprotocol->iObs[stimIdx].start[nextObs] ) {
                recording.next();
                ++iSample;
            }
            while ( iSample < reg.pprotocol->iObs[stimIdx].stop[nextObs] ) {
                recording.next();
                reg.data[stimIdx][iSample] = recording.current;
                ++iSample;
            }
            ++nextObs;
        }
    }
    return true;
}

void FitErrorPlotter::on_run_clicked()
{
    std::vector<int> protocol_indices = get_protocol_indices();
    if ( protocol_indices.empty() )
        return;

    // Adjust observations to potentially changed settings, and find longest stim duration
    int maxStimLen = 0;
    for ( int iProtocol : protocol_indices ) {
        Protocol &prot = protocols[iProtocol];
        int blankCycles = session->qGaFitterSettings().cluster_blank_after_step/prot.dt;
        for ( size_t i = 0; i < prot.stims.size(); i++ ) {
            if ( blankCycles != prot.blankCycles ) {
                std::vector<std::pair<int,int>> pairs = observeNoSteps(prot.istims[i], blankCycles);
                for ( size_t j = 0; j < pairs.size() && j < iObservations::maxObs; j++ ) {
                    prot.iObs[i].start[j] = pairs[j].first;
                    prot.iObs[i].stop[j] = pairs[j].second;
                }
                prot.blankCycles = blankCycles;
            }

            maxStimLen = std::max(maxStimLen, prot.istims[i].duration);
        }
    }

    // Find total number of target traces, and prepare list of fits & register entries
    struct RecStruct {
        std::vector<std::pair<size_t,size_t>> fit_coords;
        RegisterEntry *reg;
    };
    std::vector<RecStruct> recordings;
    int nTraces = 0;
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

    // Prepare lib
    UniversalLibrary &lib = session->project.universal();
    lib.resizeTarget(nTraces, maxStimLen);
    lib.resizeOutput(maxStimLen);

    lib.simCycles = session->qRunData().simCycles;
    lib.integrator = session->qRunData().integrator;
    lib.assignment =
            ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED | ASSIGNMENT_SUMMARY_AVERAGE
            | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_NONE | ASSIGNMENT_TIMESERIES_ZERO_UNTOUCHED_SAMPLES;

    int parameterSourceSelection = get_parameter_selection();

    // Load models into lib
    size_t modelIdx = 0;
    size_t targetOffset = 0;
    std::vector<ResultKey> keys(lib.project.expNumCandidates());
    for ( RecStruct &rec : recordings) {
        RegisterEntry &reg = *rec.reg;
        loadRecording(reg);

        for ( size_t stimIdx = 0; stimIdx < reg.pprotocol->stims.size(); stimIdx++ ) {
            // Write target traces to lib
            for ( int iSample = 0; iSample < reg.data[stimIdx].size(); iSample++ )
                lib.target[targetOffset + stimIdx + iSample*lib.targetStride] = reg.data[stimIdx][iSample];

            // Write model parameters and settings
            for ( const std::pair<size_t,size_t> fit_coords : rec.fit_coords ) {

                // Check if data already exists or is already queued
                keys[modelIdx] = ResultKey(
                            reg.pprotocol->idx,
                            stimIdx,
                            data[fit_coords.first].fits[fit_coords.second].idx,
                            parameterSourceSelection);
                if ( results.find(keys[modelIdx]) != results.end()
                     || std::find(keys.begin(), keys.begin()+modelIdx, keys[modelIdx]) != keys.begin()+modelIdx )
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
                    lib.adjustableParams[i][modelIdx] = paramValues[i];

                // Write settings
                lib.setRundata(modelIdx, reg.rund);
                lib.stim[modelIdx] = reg.pprotocol->istims[stimIdx];
                lib.obs[modelIdx] = reg.pprotocol->iObs[stimIdx];
                lib.targetOffset[modelIdx] = targetOffset + stimIdx;

                // Increment
                ++modelIdx;

                // Run a batch as soon as the model bucket is full
                if ( modelIdx == lib.project.expNumCandidates() ) {
                    push_run_pull(keys, modelIdx);
                    modelIdx = 0;
                }
            }
        }
        targetOffset += reg.pprotocol->stims.size();
    }

    // push-run-pull the remaining models
    if ( modelIdx > 0 ) {
        for ( size_t i = modelIdx; i < lib.project.expNumCandidates(); i++ ) {
            lib.stim[i].duration = 0;
            lib.iSettleDuration[i] = 0;
        }
        push_run_pull(keys, modelIdx);
    }

    replot();
}

void FitErrorPlotter::push_run_pull(std::vector<ResultKey> keys, size_t keySz)
{
    UniversalLibrary &lib = session->project.universal();
    lib.pushTarget();
    lib.push();
    lib.run();
    lib.pull(lib.summary);
    lib.pullOutput();

    for ( size_t k = 0; k < keySz; k++ ) {
        QVector<double> trace(lib.stim[k].duration);
        for ( int i = 0; i < trace.size(); i++ )
            trace[i] = lib.output[i*lib.project.expNumCandidates() + k];
        results[keys[k]] = std::make_pair(lib.summary[k], std::move(trace));
    }
}



void FitErrorPlotter::replot()
{
    std::vector<int> protocol_indices = get_protocol_indices();
    bool traces_possible = protocol_indices.size() == 1 && data.size() == 1 && data[0].fits.size() == 1;
    ui->trace_tab->setEnabled(traces_possible);

    ui->plot->clearGraphs();

    if ( !protocol_indices.empty() ) {
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
    for ( auto res : results ) {
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
            g = ui->plot->addGraph(ui->plot->xAxis, ui->plot->yAxis2);
            g->setData(keys, stim);
            g->setPen(QPen(Qt::blue));
        }
    }

    if ( ui->trace_rec->isChecked() ) {
        for ( int stimIdx : stim_indices ) {
            sizeKeys(stimIdx);
            g = ui->plot->addGraph();
            g->setData(keys, reg.data[stimIdx], true);
            g->setPen(QPen(Qt::green));
        }
    }

    if ( ui->trace_sim->isChecked() ) {
        for ( int stimIdx : stim_indices ) {
            sizeKeys(stimIdx);
            auto res = results.find(std::make_tuple(prot.idx, stimIdx, f.idx, get_parameter_selection()));
            if ( res == results.end() )
                continue;

            g = ui->plot->addGraph();
            g->setData(keys, res->second.second);
            g->setPen(QPen(Qt::red));
        }
    }

    ui->plot->rescaleAxes();
}

void FitErrorPlotter::plot_boxes(std::vector<int> protocol_indices)
{

}

std::vector<int> FitErrorPlotter::get_protocol_indices()
{
    std::vector<int> protocol_indices;
    for ( size_t i = 0; i < protocols.size(); i++ ) {
        if ( ui->protocols->item(i, 0)->checkState() == Qt::Checked ){
            protocol_indices.push_back(i);
        }
    }
    return protocol_indices;
}

int FitErrorPlotter::get_parameter_selection()
{
    int parameterSourceSelection = ui->params->currentIndex() - 2;
    if ( parameterSourceSelection == 0 )
        parameterSourceSelection = ui->epoch->value();
    return parameterSourceSelection;
}
