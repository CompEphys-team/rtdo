#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "daqfilter.h"
#include "clustering.h"

const QString GAFitter::action = QString("fit");
const quint32 GAFitter::magic = 0xadb8d269;
const quint32 GAFitter::version = 105;

GAFitter::GAFitter(Session &session) :
    SessionWorker(session),
    lib(session.project.experiment()),
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

void GAFitter::run(WaveSource src, QString VCRecord, CannedDAQ::ChannelAssociation assoc)
{
    if ( src.type != WaveSource::Deck && !session.qGaFitterSettings().useClustering )
        std::cerr << "Warning: Fitting a non-deck wave source without clustering" << std::endl;
    if ( action != this->action )
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    session.queue(actorName(), action, src.prettyName(), new Output(src, VCRecord, assoc));
}

std::vector<Stimulation> GAFitter::sanitiseDeck(std::vector<Stimulation> stimulations, bool useQueuedSettings)
{
    double dt = (useQueuedSettings ? session.qRunData().dt : session.runData().dt);
    for ( Stimulation &stim : stimulations ) {
        // Expand observation window to complete time steps
        stim.tObsBegin = floor(stim.tObsBegin / dt) * dt;
        stim.tObsEnd = ceil(stim.tObsEnd / dt) * dt;
        // Shorten stim
        stim.duration = stim.tObsEnd;
    }

    // Add a stimulation for noise sampling
    Stimulation noiseSample = stimulations.front();
    noiseSample.clear();
    noiseSample.duration = (useQueuedSettings ? session.qDaqData().varianceDuration : session.daqData().varianceDuration);
    stimulations.push_back(noiseSample);

    return stimulations;
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
    delete res;

    emit starting();

    QTime wallclock = QTime::currentTime();
    double simtime = 0;
    qT = 0;

    std::vector<Stimulation> astims = sanitiseDeck(output.stimSource.stimulations());

    daq = new DAQFilter(session);

    if ( session.daqData().simulate < 0 ) {
        daq->getCannedDAQ()->assoc = output.assoc;
        daq->getCannedDAQ()->setRecord(astims, output.VCRecord);
    }
    if ( session.daqData().simulate != 0 ) {
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
            output.targets[i] = daq->getAdjustableParam(i);
        }
    }

    output.variance = getVariance(astims.back());
    std::cout << "Baseline current noise s.d.: " << std::sqrt(output.variance) << " nA" << std::endl;

    setup(astims);

    populate();

    simtime = fit();

    // Finalise ranking and select winning parameter set
    simtime += finalise(astims);

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

void GAFitter::setup(const std::vector<Stimulation> &astims)
{
    int nParams = lib.adjustableParams.size();

    DEMethodUsed.assign(session.project.expNumCandidates()/2, 0);
    DEMethodSuccess.assign(4, 0);
    DEMethodFailed.assign(4, 0);
    DEpX.assign(session.project.expNumCandidates()/2, 0);

    bias.assign(nParams, 0);

    stims.resize(nParams);
    obsTimes.assign(nParams, {});
    errNorm.assign(nParams, 0);

    epoch = targetParam = 0;
    targetParam = findNextStim();

    if ( settings.mutationSelectivity == 2 ) {
        baseF.assign(nParams, std::vector<double>(nParams, 0));
        for ( int i = 0; i < nParams; i++ )
            baseF[i][i] = 1;
    } else {
        baseF.assign(nParams, std::vector<double>(nParams, 1));
    }

    if ( settings.useClustering ) {
        auto options = extractSeparatingClusters(constructClustersByStim(astims), nParams);
        for ( int i = 0; i < nParams; i++ ) {
            stims[i] = astims[std::get<0>(options[i])];
            for ( const Section &sec : std::get<2>(options[i]) ) {
                errNorm[i] += sec.end - sec.start;
                obsTimes[i].push_back(std::make_pair(sec.start, sec.end));
            }

            if ( settings.mutationSelectivity == 1 )
                baseF[i] = std::get<1>(options[i]);
        }
    } else {
        stims = astims;
        for ( int i = 0; i < nParams; i++ ) {
            iStimulation I(stims[i], session.runData().dt);
            errNorm[i] = I.tObsEnd - I.tObsBegin;
            obsTimes[i] = {std::make_pair(I.tObsBegin, I.tObsEnd)};

            if ( settings.mutationSelectivity == 1 ) {
                session.wavegen().diagnose(I, session.runData().dt, session.runData().simCycles);
                std::vector<Section> tObs;
                constructSections(session.wavegen().lib.diagDelta, I.tObsBegin, I.tObsEnd, nParams+1, std::vector<double>(nParams, 1),
                                  I.tObsEnd-I.tObsBegin+1, tObs);
                double norm = 0;
                for ( int j = 0; j < nParams; j++ )
                    if ( norm < fabs(tObs.front().deviations[j]) )
                        norm = fabs(tObs.front().deviations[j]);
                for ( int j = 0; j < nParams; j++ )
                    baseF[i][j] = fabs(tObs.front().deviations[j])/norm;
            }
        }
    }

    output.stims = QVector<Stimulation>::fromStdVector(stims);
    output.obsTimes.resize(nParams);
    output.baseF.resize(nParams);
    for ( int i = 0; i < nParams; i++ ) {
        for ( const std::pair<int,int> &p : obsTimes[i] )
            output.obsTimes[i].push_back(QPair<int,int>(p.first, p.second));
        output.baseF[i] = QVector<double>::fromStdVector(baseF[i]);
    }
}

double GAFitter::fit()
{
    double simtime = 0;
    for ( epoch = 0; !finished(); epoch++ ) {
        // Stimulate
        simtime += stimulate();

        // Advance
        lib.pullErr();
        if ( settings.useDE )
            procreateDE();
        else
            procreate();
        lib.push();

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
        os << out.obsTimes;
        os << out.baseF;
    }
}

void GAFitter::finish()
{
    QMutexLocker locker(&mutex);
    doFinish = true;
}

void GAFitter::load(const QString &act, const QString &, QFile &results, Result r)
{
    if ( act != action )
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    Output out(*this, r);
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
            CannedDAQ tmp(session);
            tmp.setRecord(sanitiseDeck(out.stimSource.stimulations()), out.VCRecord, false, false);
            for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
                if ( settings.constraints[i] != 3 ) // Never overwrite fixed target
                    out.targets[i] = tmp.getAdjustableParam(i);
        }
    }
    if ( ver >= 104 )
        is >> out.variance;
    if ( ver >= 105 ) {
        is >> out.stims;
        is >> out.obsTimes;
        is >> out.baseF;
    }
    m_results.push_back(std::move(out));
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
            for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ )
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
            for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
                lib.adjustableParams[j][i] = session.RNG.uniform<scalar>(min, max);
            }
        }
    }
    for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
        lib.err[i] = 0;
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
        double cost = settings.useLikelihood ? exp(output.error[epoch]) : output.error[epoch]; // Ensure bias stays positive
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
    std::vector<errTupel> p_err(lib.project.expNumCandidates());
    for ( size_t i = 0; i < p_err.size(); i++ ) {
        p_err[i].idx = i;
        p_err[i].err = lib.err[i];
        lib.err[i] = 0;
    }
    std::sort(p_err.begin(), p_err.end(), &errTupelSort);

    output.error[epoch] = settings.useLikelihood
            ? p_err[0].err // negative log likelihood
            : std::sqrt(p_err[0].err / errNorm[targetParam]); // RMSE
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
                p[p_err[i].idx] = p.multiplicative ?
                            (p[p_err[source].idx] * session.RNG.variate<scalar, std::lognormal_distribution>(
                                0, baseF[targetParam][iParam] * F * lib.adjustableParams[iParam].adjustedSigma)) :
                            session.RNG.variate<scalar, std::normal_distribution>(
                                p[p_err[source].idx], baseF[targetParam][iParam] * F * lib.adjustableParams[iParam].adjustedSigma);
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

double GAFitter::finalise(const std::vector<Stimulation> &astims)
{
    stims = astims;

    std::vector<errTupel> f_err(lib.project.expNumCandidates());
    int t = 0;
    double dt = session.runData().dt, simt = 0;

    // Evaluate existing population on all stims
    for ( targetParam = 0; targetParam < lib.adjustableParams.size() && !isAborted(); targetParam++ ) {
        obsTimes[targetParam] = observeNoSteps(iStimulation(astims[targetParam], dt), settings.cluster_blank_after_step/dt);
        for ( const std::pair<int,int> &p : obsTimes[targetParam] )
            t += p.second - p.first;
        simt += stimulate();
    }

    // Pull & sort by total cumulative error across all stims
    lib.pullErr();
    for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
        f_err[i].idx = i;
        f_err[i].err = lib.err[i];
    }
    auto winner = f_err.begin();
    std::nth_element(f_err.begin(), winner, f_err.end(), &errTupelSort);

    double err = settings.useLikelihood
            ? winner->err
            : std::sqrt(winner->err / t);
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.finalParams[i] = lib.adjustableParams[i][winner->idx];
        output.finalError[i] = err;
    }
    output.final = true;

    return simt;
}

double GAFitter::stimulate()
{
    const RunData &rd = session.runData();
    const Stimulation &aI = stims[targetParam];
    iStimulation I(aI, rd.dt);
    const std::vector<std::pair<int,int>> &obs = obsTimes[targetParam];
    auto obsIter = obs.begin();
    bool &observe = settings.useLikelihood ? lib.getLikelihood : lib.getErr;

    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.VC = true;

    // Settle library
    lib.getErr = false;
    lib.getLikelihood = false;
    lib.setVariance = false;
    lib.VClamp0 = I.baseV;
    lib.dVClamp = 0;
    lib.step(rd.settleDuration, rd.simCycles * int(rd.settleDuration/rd.dt), true);

    // Set up + settle DAQ
    daq->VC = true;
    daq->reset();
    daq->run(aI, rd.settleDuration);
    for ( size_t iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
        daq->next();

    // Set up library for stimulation
    lib.t = 0;
    lib.iT = 0;
    lib.setVariance = true; // Set variance immediately after settling
    lib.variance = output.variance;

    while ( obsIter != obs.end() && int(lib.iT) < obs.back().second ) {
        // Fast-forward library through unobserved period
        observe = false;
        int iTStep = 0;
        size_t iTDAQ = lib.iT;
        while ( (int)lib.iT < obsIter->first ) {
            getiCommandSegment(I, lib.iT, obsIter->first - lib.iT, rd.dt, lib.VClamp0, lib.dVClamp, iTStep);
            lib.step(iTStep * rd.dt, iTStep * rd.simCycles, false);
            lib.iT += iTStep;
            lib.setVariance = false;
        }
        lib.setVariance = false;

        // Fast-forward DAQ through unobserved period
        for ( ; iTDAQ < lib.iT; iTDAQ++ ) {
            daq->next();
            pushToQ(qT + iTDAQ*rd.dt, daq->voltage, daq->current, getCommandVoltage(aI, iTDAQ*rd.dt));
        }

        // Step both through observed period
        observe = true;
        while ( (int)lib.iT < obsIter->second ) {
            daq->next();
            lib.Imem = daq->current;

            // Populate VClamp0/dVClamp with the next "segment" of length tSpan = iTStep = 1
            getiCommandSegment(I, lib.iT, 1, rd.dt, lib.VClamp0, lib.dVClamp, iTStep);

            pushToQ(qT + lib.t, daq->voltage, daq->current, lib.VClamp0+lib.t*lib.dVClamp);

            lib.step();
        }

        // Advance to next observation period
        ++obsIter;
    }

    qT += lib.iT * rd.dt;
    daq->reset();

    return lib.iT * rd.dt + session.runData().settleDuration;
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

double GAFitter::getVariance(Stimulation stim)
{
    daq->reset();
    daq->run(stim, session.runData().settleDuration);
    for ( size_t iT = 0, iTEnd = session.runData().settleDuration/session.runData().dt; iT < iTEnd; iT++ )
        daq->next();

    double mean = 0, sse = 0;
    std::vector<double> samples;
    samples.reserve(daq->samplesRemaining);

    while ( daq->samplesRemaining ) {
        daq->next();
        samples.push_back(daq->current);
        mean += daq->current;
    }
    mean /= samples.size();

    for ( double sample : samples ) {
        double deviation = sample - mean;
        sse += deviation*deviation;
    }
    return sse / samples.size();
}

std::vector<std::vector<std::vector<Section>>> GAFitter::constructClustersByStim(std::vector<Stimulation> astims)
{
    double dt = session.runData().dt;
    int nParams = lib.adjustableParams.size();
    std::vector<double> norm(nParams, 1);

    std::vector<std::vector<std::vector<Section>>> clusters;
    for ( int i = 0, nStims = astims.size()-1; i < nStims; i++ ) { // nStims = size-1 to exclude noise sample added in sanitiseDeck()
        iStimulation stim(astims[i], dt);
        session.wavegen().diagnose(stim, dt, session.runData().simCycles);
        clusters.push_back(constructClusters(
                                     stim, session.wavegen().lib.diagDelta, settings.cluster_blank_after_step/dt,
                                     nParams+1, norm, settings.cluster_fragment_dur/dt, settings.cluster_threshold,
                                     settings.cluster_min_dur/settings.cluster_fragment_dur));
    }

    return clusters;
}

void GAFitter::procreateDE()
{
    std::vector<AdjustableParam> &P = lib.adjustableParams;
    int nParams = P.size();
    int nPop = session.project.expNumCandidates()/2;

    std::vector<std::vector<double>> pXList(3);
    std::vector<double> pXmed(3, 0.5);
    std::vector<double> methodCutoff(4);

    // Select the winners
    double bestErr = lib.err[0];
    int bestIdx = 0;
    for ( int i = 0; i < nPop; i++ ) {
        int iOffspring = i + nPop;
        double err = lib.err[i];
        if ( lib.err[i] > lib.err[iOffspring] ) {
            err = lib.err[iOffspring];
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

        lib.err[i] = lib.err[iOffspring] = 0;
    }

    // Populate output
    for ( int i = 0; i < nParams; i++ )
        output.params[epoch][i] = P[i][bestIdx];
    output.error[epoch] = settings.useLikelihood ? bestErr : std::sqrt(bestErr/errNorm[targetParam]);
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
