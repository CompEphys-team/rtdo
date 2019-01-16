#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "daqfilter.h"
#include "clustering.h"

const QString GAFitter::action = QString("fit");
const quint32 GAFitter::magic = 0xadb8d269;
const quint32 GAFitter::version = 107;

GAFitter::GAFitter(Session &session) :
    SessionWorker(session),
    lib(session.project.universal()),
    settings(session.gaFitterSettings()),
    qV(nullptr),
    qI(nullptr),
    qO(nullptr),
    bias(lib.adjustableParams.size(), 0),
    output(*this)
{
}

GAFitter::~GAFitter()
{
}

GAFitter::Output::Output(WaveSource stimSource, QString VCRecord, CannedDAQ::ChannelAssociation assoc) :
    stimSource(stimSource),
    VCRecord(VCRecord),
    assoc(assoc)
{}

GAFitter::Output::Output(const GAFitter &f, Result r) :
    Result(r),
    params(f.settings.maxEpochs, std::vector<scalar>(f.lib.adjustableParams.size())),
    error(f.settings.maxEpochs),
    targetParam(f.settings.maxEpochs),
    targets(f.lib.adjustableParams.size()),
    epochs(0),
    variance(0),
    final(false),
    finalParams(f.lib.adjustableParams.size()),
    finalError(f.lib.adjustableParams.size())
{
    stimSource.session =& f.session;
    for ( size_t i = 0; i < targets.size(); i++ ) // Initialise for back compat
        targets[i] = f.lib.adjustableParams.at(i).initial;
}

void GAFitter::run(WaveSource src, QString VCRecord, bool readRecConfig)
{
    if ( action != this->action )
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    if ( readRecConfig && session.qDaqData().simulate == -1 ) {
        QString cfg = VCRecord;
        cfg.replace(".atf", ".cfg");
        if ( QFileInfo(cfg).exists() )
            session.loadConfig(cfg);
    }
    session.queue(actorName(), action, src.prettyName(), new Output(src, VCRecord, session.cdaq_assoc));
}

bool GAFitter::execute(QString action, QString, Result *res, QFile &file)
{
    if ( action != this->action )
        return false;

    {
        QMutexLocker locker(&mutex);
        doFinish = false;
        aborted = false;
    }

    output = Output(*this, *res);
    output.stimSource = static_cast<Output*>(res)->stimSource;
    output.VCRecord = static_cast<Output*>(res)->VCRecord;
    output.assoc = static_cast<Output*>(res)->assoc;
    bool dryrun = res->dryrun;
    delete res;

    if ( dryrun ) {
        save(file);
        return true;
    }

    emit starting();

    QTime wallclock = QTime::currentTime();
    double simtime = 0;
    qT = 0;

    astims = output.stimSource.stimulations();

    daq = new DAQFilter(session, session.getSettings());

    if ( session.daqData().simulate < 0 ) {
        daq->getCannedDAQ()->assoc = output.assoc;
        daq->getCannedDAQ()->setRecord(astims, output.VCRecord);
        output.variance = daq->getCannedDAQ()->variance;
        std::cout << "Baseline current noise s.d.: " << std::sqrt(output.variance) << " nA" << std::endl;
    }
    if ( session.daqData().simulate != 0 ) {
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
            output.targets[i] = daq->getAdjustableParam(i);
        }
    }

    setup();

    populate();

    simtime = fit();

    // Finalise ranking and select winning parameter set
    simtime += finalise();

    double wallclockms = wallclock.msecsTo(QTime::currentTime());
    std::cout << "ms elapsed: " << wallclockms << std::endl;
    std::cout << "ms simulated: " << simtime << std::endl;
    std::cout << "Ratio: " << (simtime/wallclockms) << std::endl;

    // Finish
    output.epochs = epoch;
    m_results.push_back(output);
    emit done();

    // Save
    save(file);

    delete daq;
    daq = nullptr;

    return true;
}

void GAFitter::setup()
{
    int nParams = lib.adjustableParams.size();

    DEMethodUsed.assign(lib.NMODELS/2, 0);
    DEMethodSuccess.assign(4, 0);
    DEMethodFailed.assign(4, 0);
    DEpX.assign(lib.NMODELS/2, 0);

    bias.assign(nParams, 0);

    epoch = targetParam = 0;
    targetParam = findNextStim();

    stims.resize(nParams);
    obs.resize(nParams);
    baseF.resize(nParams, std::vector<double>(nParams, 0));

    QString obsSource = QString::fromStdString(settings.obsSource);
    if ( obsSource == Wavegen::cluster_action || obsSource == Wavegen::bubble_action ) {
        auto elites = session.wavegen().findObservations(output.stimSource.iStimulations(session.runData().dt), obsSource);
        std::vector<Stimulation> astims_ordered(nParams);
        for ( int paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
            scalar bestFitness = 0;
            size_t bestStimIdx = 0;
            for ( size_t stimIdx = 0; stimIdx < elites[paramIdx].size(); stimIdx++ ) {
                const MAPElite &el = elites[paramIdx][stimIdx];
                if ( el.fitness > bestFitness ) {
                    bestFitness = el.fitness;
                    bestStimIdx = stimIdx;
                }
            }
            stims[paramIdx] = *elites[paramIdx][bestStimIdx].wave;
            obs[paramIdx] = elites[paramIdx][bestStimIdx].obs;
            for ( int i = 0; i < nParams; i++ )
                baseF[paramIdx][i] = elites[paramIdx][bestStimIdx].deviations[i];
            astims_ordered[paramIdx] = astims[bestStimIdx];
        }
        using std::swap;
        swap(astims, astims_ordered);
    } else {
        std::vector<int> needPosthocEval;
        int paramIdx = 0;
        for ( const MAPElite &el : output.stimSource.elites() ) {
            stims[paramIdx] = *el.wave;
            obs[paramIdx] = el.obs;

            double sumDev = 0;
            for ( const scalar &dev : el.deviations )
                sumDev += dev;
            if ( sumDev == 0 ) {
                needPosthocEval.push_back(paramIdx);
            } else {
                for ( int i = 0; i < nParams; i++ )
                    baseF[paramIdx][i] = el.deviations[i];
            }
            ++paramIdx;
        }

        if ( !needPosthocEval.empty() ) {
            std::vector<MAPElite> posthoc = session.wavegen().evaluatePremade(stims, obs);
            for ( int paramIdx : needPosthocEval )
                for ( int i = 0; i < nParams; i++ )
                    baseF[paramIdx][i] = posthoc[paramIdx].deviations[i];
        }
    }

    if ( settings.mutationSelectivity == 2 ) {
        baseF.assign(nParams, std::vector<double>(nParams, 0));
        for ( int i = 0; i < nParams; i++ )
            baseF[i][i] = 1;
    } else if ( settings.mutationSelectivity == 0 ) {
        baseF.assign(nParams, std::vector<double>(nParams, 1));
    }

    errNorm.resize(nParams);
    for ( int paramIdx = 0; paramIdx < nParams; paramIdx++ )
        errNorm[paramIdx] = obs[paramIdx].duration();

    output.stims = QVector<iStimulation>::fromStdVector(stims);
    output.obs = QVector<iObservations>::fromStdVector(obs);
    output.baseF.resize(nParams);
    for ( int i = 0; i < nParams; i++ )
        output.baseF[i] = QVector<double>::fromStdVector(baseF[i]);
}

double GAFitter::fit()
{
    double simtime = 0;
    for ( epoch = 0; !finished(); epoch++ ) {
        // Stimulate
        simtime += stimulate();

        // Advance
        lib.pull(lib.summary);
        if ( settings.useDE )
            procreateDE();
        else
            procreate();

        emit progress(epoch);
    }

    return simtime;
}

void GAFitter::save(QFile &file)
{
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) ) {
        const Output &out = m_results.back();
        os << out.stimSource << out.epochs;
        for ( quint32 i = 0; i < out.epochs; i++ ) {
            os << out.targetParam[i] << out.error[i];
            for ( const scalar &p : out.params[i] )
                os << p;
        }
        for ( const scalar &t : out.targets )
            os << t;
        os << out.final;
        for ( const scalar &p : out.finalParams )
            os << p;
        for ( const scalar &e : out.finalError )
            os << e;
        os << out.VCRecord;
        os << out.variance;
        os << out.stims;
        os << out.obs;
        os << out.baseF;
        os << qint32(out.assoc.Iidx) << qint32(out.assoc.Vidx) << qint32(out.assoc.V2idx);
        os << out.assoc.Iscale << out.assoc.Vscale << out.assoc.V2scale;
    }
}

void GAFitter::finish()
{
    QMutexLocker locker(&mutex);
    doFinish = true;
}

Result *GAFitter::load(const QString &act, const QString &, QFile &results, Result r)
{
    if ( act != action )
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    Output *p_out;
    if ( r.dryrun ) {
        p_out = new Output(*this, r);
    } else {
        m_results.emplace_back(*this, r);
        p_out =& m_results.back();
    }
    Output &out = *p_out;

    is >> out.stimSource >> out.epochs;
    out.targetParam.resize(out.epochs);
    out.params.resize(out.epochs);
    out.error.resize(out.epochs);
    for ( quint32 i = 0; i < out.epochs; i++ ) {
        is >> out.targetParam[i] >> out.error[i];
        for ( scalar &p : out.params[i] )
            is >> p;
    }
    if ( ver >= 101 )
        for ( scalar &t : out.targets )
            is >> t;
    if ( ver >= 102 ) {
        is >> out.final;
        for ( scalar &p : out.finalParams )
            is >> p;
        for ( scalar &e : out.finalError )
            is >> e;
    }
    if ( ver >= 103 ) {
        is >> out.VCRecord;
        if ( session.daqData().simulate < 0 ) {
            CannedDAQ tmp(session, session.getSettings());
            tmp.setRecord(out.stimSource.stimulations(), out.VCRecord, false);
            for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
                if ( settings.constraints[i] != 3 ) // Never overwrite fixed target
                    out.targets[i] = tmp.getAdjustableParam(i);
        }
    }
    if ( ver >= 104 )
        is >> out.variance;
    if ( ver >= 105 ) {
        if ( ver < 107 ) {
            QVector<Stimulation> stims;
            QVector<QVector<QPair<int,int>>> obsTimes;
            is >> stims >> obsTimes;
            for ( const Stimulation &I : stims )
                out.stims.push_back(iStimulation(I, session.runData().dt));
            for ( QVector<QPair<int,int>> obst : obsTimes ) {
                out.obs.push_back({{},{}});
                for ( int i = 0; i < obst.size(); i++ ) {
                    if ( i < int(iObservations::maxObs) ) {
                        out.obs.back().start[i] = obst[i].first;
                        out.obs.back().stop[i] = obst[i].second;
                    }
                }
            }
        } else {
            is >> out.stims;
            is >> out.obs;
        }
        is >> out.baseF;
    }
    if ( ver >= 106 ) {
        qint32 I, V, V2;
        is >> I >> V >> V2;
        out.assoc.Iidx = I;
        out.assoc.Vidx = V;
        out.assoc.V2idx = V2;
        is >> out.assoc.Iscale >> out.assoc.Vscale >> out.assoc.V2scale;
    }

    return p_out;
}

bool GAFitter::finished()
{
    QMutexLocker locker(&mutex);
    return aborted || doFinish || epoch >= settings.maxEpochs;
}

void GAFitter::populate()
{
    for ( size_t j = 0; j < lib.adjustableParams.size(); j++ ) {
        if ( settings.constraints[j] == 2 || settings.constraints[j] == 3 ) { // Fixed value
            scalar value = settings.constraints[j] == 2 ? settings.fixedValue[j] : output.targets[j];
            for ( size_t i = 0; i < lib.NMODELS; i++ )
                lib.adjustableParams[j][i] = value;
        } else {
            scalar min, max;
            if ( settings.constraints[j] == 0 ) {
                min = lib.adjustableParams[j].min;
                max = lib.adjustableParams[j].max;
            } else {
                min = settings.min[j];
                max = settings.max[j];
            }
            for ( size_t i = 0; i < lib.NMODELS; i++ ) {
                lib.adjustableParams[j][i] = session.RNG.uniform<scalar>(min, max);
            }
        }
    }
    lib.push();
}

quint32 GAFitter::findNextStim()
{
    // Exclude stims with fixed parameter value (constraints==2)
    // translate stimIdx to point to a contiguous array of the actually used stimulations
    quint32 nStims = 0, previousStimIdx = 0;
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        if ( settings.constraints[i] < 2 ) {
            ++nStims;
            if ( i < targetParam )
                ++previousStimIdx;
        }
    }

    quint32 nextStimIdx(previousStimIdx);
    if ( settings.randomOrder == 3 ) { // Sequence-biased random
        // bias[i] == number of epochs since i was last used
        if ( epoch == 0 ) {
            for ( size_t i = 0; i < nStims; i++ )
                bias[i] = 1;
        } else {
            bias[previousStimIdx] = 0;
            for ( size_t i = 0; i < nStims; i++ )
                ++bias[i];
        }
        std::vector<int> cumBias(nStims, bias[0]);
        for ( size_t i = 1; i < nStims; i++ ) // Cumulative sum
            cumBias[i] = cumBias[i-1] + bias[i];
        int choice = session.RNG.uniform<int>(0, cumBias.back()-1);
        for ( nextStimIdx = 0; choice >= cumBias[nextStimIdx]; nextStimIdx++ ) ;
    } else if ( settings.randomOrder == 2 ) { // Error-biased random
        double cost = output.error[epoch];
        if ( epoch == previousStimIdx ) // Initial: Full error
            bias[previousStimIdx] = cost;
        else // Recursively decay bias according to settings
            bias[previousStimIdx] = settings.orderBiasDecay * cost + (1-settings.orderBiasDecay) * bias[previousStimIdx];

        if ( epoch + 1 < nStims ) { // Initial round: Sequential order
            nextStimIdx = previousStimIdx + 1;
        } else if ( int(epoch) < settings.orderBiasStartEpoch ) { // Further unbiased rounds: Random order
            nextStimIdx = session.RNG.uniform<quint32>(0, nStims-1);
        } else { // Biased rounds
            double sumBias = 0;
            for ( size_t i = 0; i < nStims; i++ )
                sumBias += bias[i];
            double choice = session.RNG.uniform(0.0, sumBias);
            for ( size_t i = 0; i < nStims; i++ ) {
                choice -= bias[i];
                if ( choice < 0 ) {
                    nextStimIdx = i;
                    break;
                }
            }
        }
    } else if ( settings.randomOrder == 1 )
        nextStimIdx = session.RNG.uniform<quint32>(0, nStims-1);
    else
        nextStimIdx = (previousStimIdx + 1) % nStims;

    // Translate nextStimIdx back to index into full stim array
    for ( size_t i = 0; i <= nextStimIdx; i++ )
        if ( settings.constraints[i] >= 2 )
            ++nextStimIdx;

    return nextStimIdx;
}

bool GAFitter::errTupelSort(const errTupel &x, const errTupel &y)
{
    // make sure NaN is largest
    if ( std::isnan(x.err) ) return false;
    if ( std::isnan(y.err) ) return true;
    return x.err < y.err;
}

void GAFitter::procreate()
{
    std::vector<errTupel> p_err(lib.NMODELS);
    for ( size_t i = 0; i < p_err.size(); i++ ) {
        p_err[i].idx = i;
        p_err[i].err = lib.summary[i];
    }
    std::sort(p_err.begin(), p_err.end(), &errTupelSort);

    output.error[epoch] = std::sqrt(p_err[0].err / errNorm[targetParam]); // RMSE
    output.targetParam[epoch] = targetParam;
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.params[epoch][i] = lib.adjustableParams[i][p_err[0].idx];
    }

    targetParam = findNextStim();

    double F = settings.decaySigma ? settings.sigmaInitial * std::exp2(-double(epoch)/settings.sigmaHalflife) : 1;

    // Mutate
    for ( size_t i = p_err.size()-settings.nReinit-1; i >= settings.nElite; i-- ) {
        // Bias reproductions towards the elite in roughly linear fashion by restricting the choice range
        size_t targetSource = session.RNG.uniform<size_t>(0, i-settings.nElite);
        size_t source = targetSource;

        size_t otherSource = targetSource;
        unsigned int sourceSelect = 0;
        if ( settings.crossover > 0 && session.RNG.uniform<double>(0,1) < settings.crossover )
            otherSource = session.RNG.uniform<size_t>(0, i-settings.nElite);

        for ( size_t iParam = 0; iParam < lib.adjustableParams.size(); iParam++ ) {

            // Ignore fixed-value parameters
            if ( settings.constraints[iParam] >= 2 )
                continue;

            // Parameter-wise crossover, implemented with minimal RNG use
            if ( targetSource != otherSource ) {
                if ( iParam % (8*sizeof sourceSelect) == 0 )
                    sourceSelect = session.RNG.uniform<unsigned int>(0, ~0);
                source = (sourceSelect & 0x1) ? otherSource : targetSource;
                sourceSelect = sourceSelect >> 1;
            }

            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( settings.mutationSelectivity < 2 || iParam == targetParam ) {
                // Mutate target param
                scalar sigma = settings.constraints[iParam] == 1 ? settings.sigma[iParam] : lib.adjustableParams[iParam].sigma;
                if ( !p.multiplicative )
                    p[p_err[i].idx] = session.RNG.variate<scalar, std::normal_distribution>(
                                        p[p_err[source].idx], baseF[targetParam][iParam] * F * sigma);
                else if ( iParam < lib.model.nNormalAdjustableParams ) {
                    scalar factor = -1;
                    while ( factor < 0 )
                        factor = session.RNG.variate<scalar, std::normal_distribution>(
                                  1, baseF[targetParam][iParam] * F * sigma);
                    p[p_err[i].idx] = p[p_err[source].idx] * factor;
                } else {
                    p[p_err[i].idx] = p[p_err[source].idx] *
                            ( session.RNG.variate<scalar, std::uniform_real_distribution>(0, settings.decaySigma ? 2*F/settings.sigmaInitial : 2) < baseF[targetParam][iParam] ? -1 : 1 );
                }
                if ( settings.constraints[iParam] == 0 ) {
                    if ( p[p_err[i].idx] < p.min )
                        p[p_err[i].idx] = p.min;
                    if ( p[p_err[i].idx] > p.max )
                        p[p_err[i].idx] = p.max;
                } else {
                    if ( p[p_err[i].idx] < settings.min[iParam] )
                        p[p_err[i].idx] = settings.min[iParam];
                    if ( p[p_err[i].idx] > settings.max[iParam] )
                        p[p_err[i].idx] = settings.max[iParam];
                }
            } else {
                // Copy non-target params
                p[p_err[i].idx] = p[p_err[source].idx];
            }
        }
    }

    // Reinit
    for ( size_t source = 0; source < settings.nReinit; source++ ) {
        size_t i = p_err.size() - source - 1;
        for ( size_t iParam = 0; iParam < lib.adjustableParams.size(); iParam++ ) {
            if ( settings.constraints[iParam] >= 2 )
                continue;
            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( iParam == targetParam ) {
                if ( settings.constraints[iParam] == 0 )
                    p[p_err[i].idx] = session.RNG.uniform(p.min, p.max);
                else
                    p[p_err[i].idx] = session.RNG.uniform(settings.min[iParam], settings.max[iParam]);
            } else {
                p[p_err[i].idx] = p[p_err[source].idx];
            }
        }
    }
}

double GAFitter::finalise()
{
    std::vector<errTupel> f_err(lib.NMODELS);
    int t = 0;
    double dt = session.runData().dt, simt = 0;

    // restore original stims as specified in source
    astims = output.stimSource.stimulations();
    stims = output.stimSource.iStimulations(dt);

    // Evaluate existing population on all stims
    for ( targetParam = 0; targetParam < stims.size() && !isAborted(); targetParam++ ) {
        obs[targetParam] = iObserveNoSteps(stims[targetParam], session.wavegenData().cluster.blank/dt);
        t += obs[targetParam].duration();
        simt += stimulate(targetParam>0 ? ASSIGNMENT_SUMMARY_PERSIST : 0);
    }

    // Pull & sort by total cumulative error across all stims
    lib.pull(lib.summary);
    for ( size_t i = 0; i < lib.NMODELS; i++ ) {
        f_err[i].idx = i;
        f_err[i].err = lib.summary[i];
    }
    auto winner = f_err.begin();
    std::nth_element(f_err.begin(), winner, f_err.end(), &errTupelSort);

    double err = std::sqrt(winner->err / t);
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.finalParams[i] = lib.adjustableParams[i][winner->idx];
        output.finalError[i] = err;
    }
    output.final = true;

    return simt;
}

double GAFitter::stimulate(unsigned int extra_assignments)
{
    const RunData &rd = session.runData();
    const Stimulation &aI = astims[targetParam];
    iStimulation I = stims[targetParam];

    // Set up library
    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.setRundata(0, rd);

    lib.setSingularStim();
    lib.stim[0] = I;
    lib.obs[0] = obs[targetParam];

    lib.setSingularTarget();
    lib.resizeTarget(1, I.duration);
    lib.targetOffset[0] = 0;

    lib.assignment = lib.assignment_base | extra_assignments
            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED;
    lib.push();

    // Set up + settle DAQ
    daq->VC = true;
    daq->reset();
    daq->run(aI, rd.settleDuration);
    for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
        daq->next();

    // Step DAQ through full stimulation
    for ( int iT = 0; iT < I.duration; iT++ ) {
        daq->next();
        pushToQ(qT + iT*rd.dt, daq->voltage, daq->current, getCommandVoltage(aI, iT*rd.dt));
        lib.target[iT] = daq->current;
    }
    daq->reset();

    // Run lib against target
    lib.pushTarget();
    lib.run();

    qT += I.duration * rd.dt;

    return I.duration * rd.dt + rd.settleDuration;
}

void GAFitter::pushToQ(double t, double V, double I, double O)
{
    if ( qV )
        qV->push({t,V});
    if ( qI )
        qI->push({t,I});
    if ( qO )
        qO->push({t,O});
}

void GAFitter::procreateDE()
{
    std::vector<AdjustableParam> &P = lib.adjustableParams;
    int nParams = P.size();
    int nPop = lib.NMODELS/2;

    std::vector<std::vector<double>> pXList(3);
    std::vector<double> pXmed(3, 0.5);
    std::vector<double> methodCutoff(4);

    // Select the winners
    double bestErr = lib.summary[0];
    int bestIdx = 0;
    for ( int i = 0; i < nPop; i++ ) {
        int iOffspring = i + nPop;
        double err = lib.summary[i];
        if ( lib.summary[i] > lib.summary[iOffspring] ) {
            err = lib.summary[iOffspring];
            for ( int j = 0; j < nParams; j++ ) // replace parent with more successful offspring
                P[j][i] = P[j][iOffspring];
            ++DEMethodSuccess[DEMethodUsed[i]];
        } else {
            for ( int j = 0; j < nParams; j++ ) // replace failed offspring with parent, ready for mutation
                P[j][iOffspring] = P[j][i];
            ++DEMethodFailed[DEMethodUsed[i]];
            if ( DEMethodUsed[i] < 3 )
                pXList[DEMethodUsed[i]].push_back(DEpX[i]);
        }

        if ( err < bestErr ) {
            bestIdx = i;
            bestErr = err;
        }

        lib.summary[i] = lib.summary[iOffspring] = 0;
    }

    // Populate output
    for ( int i = 0; i < nParams; i++ )
        output.params[epoch][i] = P[i][bestIdx];
    output.error[epoch] = std::sqrt(bestErr/errNorm[targetParam]);
    output.targetParam[epoch] = targetParam;

    // Select new target
    targetParam = findNextStim();

    // Calculate mutation probabilities
    double successRateTotal = 0;
    if ( epoch == 0 ) {
        for ( int i = 0; i < 4; i++ ) {
            successRateTotal += 0.25;
            methodCutoff[i] = successRateTotal;
        }
    } else {
        for ( int i = 0; i < 4; i++ ) {
            double successRate = DEMethodSuccess[i] / (DEMethodSuccess[i] + DEMethodFailed[i]) + 0.01;
            successRateTotal += successRate;
            methodCutoff[i] = successRateTotal;

            if ( i < 3 ) {
                auto nth = pXList[i].begin() + pXList[i].size()/2;
                std::nth_element(pXList[i].begin(), nth, pXList[i].end());
                if ( pXList[i].size() % 2 )
                    pXmed[i] = *nth;
                else
                    pXmed[i] = (*nth + *std::max_element(pXList[i].begin(), nth))/2;
                pXList[i].clear();
            }
        }
    }

    // Procreate
    for ( int i = 0; i < nPop; i++ ) {
        double method = session.RNG.uniform(0., successRateTotal);
        double F = session.RNG.variate<double>(0.5, 0.3);
        int r1, r2, r3, r4, r5, forcedJ;
        do { r1 = session.RNG.uniform<int>(0, nPop-1); } while ( r1 == i );
        do { r2 = session.RNG.uniform<int>(0, nPop-1); } while ( r2 == i || r2 == r1 );
        do { r3 = session.RNG.uniform<int>(0, nPop-1); } while ( r3 == i || r3 == r1 || r3 == r2 );
        do { forcedJ = session.RNG.uniform<int>(0, nParams-1); } while ( settings.constraints[forcedJ] >= 2 );

        if ( method < methodCutoff[0] ) {
        // rand/1/bin
            DEMethodUsed[i] = 0;
            DEpX[i] = session.RNG.variate<double>(pXmed[0], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][r1] + F * baseF[targetParam][j] * (P[j][r2] - P[j][r3]);
            }
        } else if ( method < methodCutoff[1] ) {
        // rand-to-best/2/bin
            DEMethodUsed[i] = 1;
            do { r4 = session.RNG.uniform<int>(0, nPop-1); } while ( r4 == i || r4 == r1 || r4 == r2 || r4 == r3 );
            DEpX[i] = session.RNG.variate<double>(pXmed[1], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][i] + F * baseF[targetParam][j] * (P[j][bestIdx] - P[j][i] + P[j][r1] - P[j][r2] + P[j][r3] - P[j][r4]);
            }
        } else if ( method < methodCutoff[2] ) {
        // rand/2/bin
            DEMethodUsed[i] = 2;
            do { r4 = session.RNG.uniform<int>(0, nPop-1); } while ( r4 == i || r4 == r1 || r4 == r2 || r4 == r3 );
            do { r5 = session.RNG.uniform<int>(0, nPop-1); } while ( r5 == i || r5 == r1 || r5 == r2 || r5 == r3 || r5 == r4 );
            DEpX[i] = session.RNG.variate<double>(pXmed[2], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][r1] + F * baseF[targetParam][j] * (P[j][r2] - P[j][r3] + P[j][r4] - P[j][r5]);
            }
        } else {
        // current-to-rand/1
            DEMethodUsed[i] = 3;
            double K = session.RNG.uniform(0.,1.);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 )
                    P[j][i + nPop] = P[j][i] + baseF[targetParam][j] * K * ((P[j][r1] - P[j][i]) + F*(P[j][r2] - P[j][r3]));
            }
        }

        // Apply limits
        for ( int j = 0; j < nParams; j++ ) {
            if ( settings.constraints[j] == 0 ) {
                if ( P[j][i + nPop] < P[j].min )
                    P[j][i + nPop] = P[j].min;
                if ( P[j][i + nPop] > P[j].max )
                    P[j][i + nPop] = P[j].max;
            } else {
                if ( P[j][i + nPop] < settings.min[j] )
                    P[j][i + nPop] = settings.min[j];
                if ( P[j][i + nPop] > settings.max[j] )
                    P[j][i + nPop] = settings.max[j];
            }
        }
    }
}
