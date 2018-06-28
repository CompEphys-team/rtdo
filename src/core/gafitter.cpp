#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "daqfilter.h"

const QString GAFitter::action = QString("fit");
const quint32 GAFitter::magic = 0xadb8d269;
const quint32 GAFitter::version = 104;

GAFitter::GAFitter(Session &session) :
    SessionWorker(session),
    lib(session.project.experiment()),
    settings(session.gaFitterSettings()),
    qV(nullptr),
    qI(nullptr),
    qO(nullptr),
    bias(lib.adjustableParams.size(), 0),
    p_err(lib.project.expNumCandidates()),
    output(*this)
{
}

GAFitter::~GAFitter()
{
}

GAFitter::Output::Output(WaveSource deck, QString VCRecord) :
    deck(deck),
    VCRecord(VCRecord)
{}

GAFitter::Output::Output(const GAFitter &f, Result r) :
    Result(r),
    params(f.settings.maxEpochs, std::vector<scalar>(f.lib.adjustableParams.size())),
    error(f.settings.maxEpochs),
    stimIdx(f.settings.maxEpochs),
    targets(f.lib.adjustableParams.size()),
    epochs(0),
    variance(0),
    final(false),
    finalParams(f.lib.adjustableParams.size()),
    finalError(f.lib.adjustableParams.size())
{
    deck.session =& f.session;
    for ( size_t i = 0; i < targets.size(); i++ ) // Initialise for back compat
        targets[i] = f.lib.adjustableParams.at(i).initial;
}

void GAFitter::run(WaveSource src, QString VCRecord)
{
    if ( src.type != WaveSource::Deck )
        throw std::runtime_error("Wave source for GAFitter must be a deck.");
    session.queue(actorName(), action, QString("Deck %1").arg(src.idx), new Output(src, VCRecord));
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
    output.deck = static_cast<Output*>(res)->deck;
    output.VCRecord = static_cast<Output*>(res)->VCRecord;
    delete res;

    emit starting();

    QTime wallclock = QTime::currentTime();
    double simtime = 0;
    qT = 0;

    // Prepare
    astims = sanitiseDeck(output.deck.stimulations());
    stims.clear();
    for ( Stimulation stim : astims )
        stims.push_back(iStimulation(stim, session.runData().dt));
    stimIdx = 0;

    daq = new DAQFilter(session);

    if ( session.daqData().simulate < 0 )
        daq->getCannedDAQ()->setRecord(astims, output.VCRecord);
    if ( session.daqData().simulate != 0 ) {
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
            output.targets[i] = daq->getAdjustableParam(i);
        }
    }

    output.variance = getVariance();
    std::cout << "Baseline current noise s.d.: " << std::sqrt(output.variance) << " nA" << std::endl;

    // Fit
    populate();
    for ( epoch = 0; !finished(); epoch++ ) {
        // Stimulate
        stimulate();
        simtime += astims.at(stimIdx).duration + session.runData().settleDuration;

        // Advance
        lib.pullErr();
        procreate();
        lib.push();

        emit progress(epoch);
    }

    double wallclockms = wallclock.msecsTo(QTime::currentTime());
    std::cout << "ms elapsed: " << wallclockms << std::endl;
    std::cout << "ms simulated: " << simtime << std::endl;
    std::cout << "Ratio: " << (simtime/wallclockms) << std::endl;

    // Finalise ranking and select winning parameter set
    finalise();

    // Finish
    output.epochs = epoch;
    m_results.push_back(output);
    emit done();

    // Save
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) ) {
        const Output &out = m_results.back();
        os << out.deck << out.epochs;
        for ( quint32 i = 0; i < out.epochs; i++ ) {
            os << out.stimIdx[i] << out.error[i];
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
    }

    delete daq;
    daq = nullptr;

    return true;
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
    is >> out.deck >> out.epochs;
    out.stimIdx.resize(out.epochs);
    out.params.resize(out.epochs);
    out.error.resize(out.epochs);
    for ( quint32 i = 0; i < out.epochs; i++ ) {
        is >> out.stimIdx[i] >> out.error[i];
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
            tmp.setRecord(sanitiseDeck(out.deck.stimulations()), out.VCRecord, false, false);
            for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
                if ( settings.constraints[i] != 3 ) // Never overwrite fixed target
                    out.targets[i] = tmp.getAdjustableParam(i);
        }
    }
    if ( ver >= 104 )
        is >> out.variance;
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
            if ( i < stimIdx )
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
        double cost = settings.useLikelihood ? exp(p_err[0].err) : p_err[0].err; // Ensure bias stays positive
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
    for ( size_t i = 0; i < p_err.size(); i++ ) {
        p_err[i].idx = i;
        p_err[i].err = lib.err[i];
        lib.err[i] = 0;
    }
    std::sort(p_err.begin(), p_err.end(), &errTupelSort);

    output.error[epoch] = settings.useLikelihood
            ? p_err[0].err // negative log likelihood
            : std::sqrt(p_err[0].err / (stims.at(stimIdx).tObsEnd-stims.at(stimIdx).tObsBegin)); // RMSE
    output.stimIdx[epoch] = stimIdx;
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.params[epoch][i] = lib.adjustableParams[i][p_err[0].idx];
    }

    do {
        stimIdx = findNextStim();
        // In cached use, searching for an available stim makes sense...
        // In uncached use, this would just cause a busy loop and waste time that could be used to mutate/reinit
        if ( !session.daqData().cache.active )
            break;
    } while ( daq->throttledFor(astims.at(stimIdx)) > 0 );

    scalar sigma = lib.adjustableParams[stimIdx].adjustedSigma;
    if ( settings.decaySigma )
        sigma = settings.sigmaInitial * sigma * std::exp2(-double(epoch)/settings.sigmaHalflife);

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
            if ( iParam == stimIdx ) {
                // Mutate target param
                p[p_err[i].idx] = p.multiplicative ?
                            (p[p_err[source].idx] * session.RNG.variate<scalar, std::lognormal_distribution>(0, sigma)) :
                            session.RNG.variate<scalar, std::normal_distribution>(p[p_err[source].idx], sigma);
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
            if ( iParam == stimIdx ) {
                if ( settings.constraints[iParam] == 0 )
                    p[p_err[i].idx] = session.RNG.uniform(p.min, p.max);
                else
                    p[p_err[i].idx] = session.RNG.uniform(settings.min[iParam], settings.max[iParam]);
            } else {
                p[p_err[i].idx] = p[p_err[source].idx];
            }
        }
    }

    if ( !session.daqData().cache.active )
        QThread::msleep(daq->throttledFor(astims.at(stimIdx)));
}

void GAFitter::finalise()
{
    std::vector<std::vector<errTupel>> f_err(lib.adjustableParams.size(), std::vector<errTupel>(lib.project.expNumCandidates()));

    // Evaluate existing population on all stims
    for ( stimIdx = 0; stimIdx < lib.adjustableParams.size() && !isAborted(); stimIdx++ ) {
        if ( settings.constraints[stimIdx] >= 2 )
            continue;

        // Stimulate
        stimulate();

        // Gather and reset error
        lib.pullErr();
        for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
            f_err[stimIdx][i].idx = i;
            f_err[stimIdx][i].err = lib.err[i];
            lib.err[i] = 0;
        }
        lib.pushErr();

        // Sort
        std::sort(f_err[stimIdx].begin(), f_err[stimIdx].end(), &errTupelSort);
    }

    // Select final
    std::vector<errTupel> sumRank(lib.project.expNumCandidates());
    // Abusing errTupel (idx, err) as (idx, sumRank).
    for ( size_t i = 0; i < sumRank.size(); i++ ) {
        sumRank[i].idx = i;
        sumRank[i].err = 0;
    }
    // For each parameter set, add up the ranking across all stims
    for ( const std::vector<errTupel> &ranked : f_err ) {
        for ( size_t i = 0; i < sumRank.size(); i++ ) {
            if ( settings.constraints[i] < 2 )
                sumRank[ranked[i].idx].err += i;
        }
    }
    // Sort by sumRank, ascending
    std::sort(sumRank.begin(), sumRank.end(), &errTupelSort);

    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.finalParams[i] = lib.adjustableParams[i][sumRank[0].idx];
        for ( size_t j = 0; j < sumRank.size(); j++ ) {
            if ( f_err[i][j].idx == sumRank[0].idx ) {
                output.finalError[i] = settings.useLikelihood
                        ? f_err[i][j].err
                        : std::sqrt(f_err[i][j].err / (stims.at(i).tObsEnd-stims.at(i).tObsBegin));
                break;
            }
        }
    }
    output.final = true;
}

void GAFitter::stimulate()
{
    const iStimulation &I = stims.at(stimIdx);
    const Stimulation &aI = astims.at(stimIdx);
    bool &observe = settings.useLikelihood ? lib.getLikelihood : lib.getErr;
    const RunData &rd = session.runData();

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

    // Fast-forward library through unobserved period
    lib.t = 0;
    lib.iT = 0;
    lib.setVariance = true; // Set variance immediately after settling
    lib.variance = output.variance;
    observe = false;
    int iTStep = 0;
    while ( (int)lib.iT < I.tObsBegin ) {
        getiCommandSegment(I, lib.iT, I.tObsBegin - lib.iT, rd.dt, lib.VClamp0, lib.dVClamp, iTStep);
        lib.step(iTStep * rd.dt, iTStep * rd.simCycles, false);
        lib.iT += iTStep;
        lib.setVariance = false;
    }

    // Set up + settle DAQ
    daq->VC = true;
    daq->reset();
    daq->run(aI, rd.settleDuration);
    for ( size_t iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
        daq->next();

    // Fast-forward DAQ through unobserved period
    for ( size_t iT = 0; iT < lib.iT; iT++ ) {
        daq->next();
        pushToQ(qT + iT*rd.dt, daq->voltage, daq->current, getCommandVoltage(aI, iT*rd.dt));
    }

    // Step both through observed period -- assumes a properly truncated stim (tObsEnd==duration)
    observe = true;
    while ( (int)lib.iT < I.duration ) {
        daq->next();
        lib.Imem = daq->current;

        // Populate VClamp0/dVClamp with the next "segment" of length tSpan = iTStep = 1
        getiCommandSegment(I, lib.iT, 1, rd.dt, lib.VClamp0, lib.dVClamp, iTStep);

        pushToQ(qT + lib.t, daq->voltage, daq->current, lib.VClamp0+lib.t*lib.dVClamp);

        lib.step();
        lib.setVariance = false; // In case tObsBegin==0
    }

    qT += aI.duration;
    daq->reset();
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

double GAFitter::getVariance()
{
    daq->reset();
    daq->run(astims.back(), session.runData().settleDuration);
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
