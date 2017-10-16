#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "daqfilter.h"

const QString GAFitter::action = QString("fit");
const quint32 GAFitter::magic = 0xadb8d269;
const quint32 GAFitter::version = 102;

GAFitter::GAFitter(Session &session) :
    SessionWorker(session),
    lib(session.project.experiment()),
    settings(session.gaFitterSettings()),
    qV(nullptr),
    qI(nullptr),
    qO(nullptr),
    aborted(false),
    bias(lib.adjustableParams.size(), 0),
    p_err(lib.project.expNumCandidates()),
    output(*this)
{
    connect(this, &GAFitter::didAbort, this, &GAFitter::clearAbort);
}

GAFitter::~GAFitter()
{
}

void GAFitter::abort()
{
    aborted = true;
    emit didAbort();
}

void GAFitter::clearAbort()
{
    aborted = false;
}

GAFitter::Output::Output(const GAFitter &f, Result r) :
    Result(r),
    params(f.settings.maxEpochs, std::vector<scalar>(f.lib.adjustableParams.size())),
    error(f.settings.maxEpochs),
    stimIdx(f.settings.maxEpochs),
    targets(f.lib.adjustableParams.size()),
    epochs(0),
    settings(f.settings),
    daqSettings(f.session.daqData()),
    final(false),
    finalParams(f.lib.adjustableParams.size()),
    finalError(f.lib.adjustableParams.size())
{
    deck.session =& f.session;
    for ( size_t i = 0; i < targets.size(); i++ ) // Initialise for back compat
        targets[i] = f.lib.adjustableParams.at(i).initial;
}

void GAFitter::run(WaveSource src)
{
    if ( aborted )
        return;
    if ( src.type != WaveSource::Deck )
        throw std::runtime_error("Wave source for GAFitter must be a deck.");

    emit starting();

    QTime wallclock = QTime::currentTime();
    double simtime = 0;
    qT = 0;

    // Prepare
    output = Output(*this);
    output.deck = std::move(src);
    stims = output.deck.stimulations();
    // Integrate settling into all stimulations
    double settleDuration = session.runData().settleDuration;
    for ( Stimulation &stim : stims ) {
        stim.duration += settleDuration;
        // Expand observation window to complete time steps (for tObsEnd this happens for free)
        stim.tObsBegin = floor((stim.tObsBegin+settleDuration) / session.project.dt()) * session.project.dt();
        stim.tObsEnd += settleDuration;
        for ( Stimulation::Step &step : stim ) {
            step.t += settleDuration;
        }
        if ( stim.begin()->ramp )
            stim.insert(stim.begin(), Stimulation::Step {(scalar)settleDuration, stim.baseV, false});
    }
    stimIdx = 0;
    doFinish = false;

    daq = new DAQFilter(session);

    if ( session.daqData().simulate ) {
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
            output.targets[i] = daq->getAdjustableParam(i);
        }
    }

    // Fit
    populate();
    for ( epoch = 0; !aborted && !finished(); epoch++ ) {
        const Stimulation &stim = stims.at(stimIdx);

        // Stimulate
        stimulate(stim);
        simtime += stim.duration;

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
    QFile file(session.log(this, action, m_results.back()));
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
    }

    delete daq;
    daq = nullptr;
}

void GAFitter::finish()
{
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
    m_results.push_back(std::move(out));
}

bool GAFitter::finished()
{
    return doFinish || epoch >= settings.maxEpochs;
}

void GAFitter::populate()
{
    for ( AdjustableParam &p : lib.adjustableParams ) {
        for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
            p[i] = session.RNG.uniform<scalar>(p.min, p.max);
        }
    }
    for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
        lib.err[i] = 0;
    }
    lib.push();
}

quint32 GAFitter::findNextStim()
{
    quint32 nextStimIdx(stimIdx);
    if ( settings.randomOrder == 2 ) {
        if ( epoch == stimIdx ) // Initial: Full error
            bias[stimIdx] = p_err[0].err;
        else // Recursively decay bias according to settings
            bias[stimIdx] = settings.orderBiasDecay * p_err[0].err + (1-settings.orderBiasDecay) * bias[stimIdx];

        if ( epoch + 1 < bias.size() ) { // Initial round: Sequential order
            nextStimIdx = stimIdx + 1;
        } else if ( int(epoch) < settings.orderBiasStartEpoch ) { // Further unbiased rounds: Random order
            nextStimIdx = session.RNG.uniform<quint32>(0, stims.size()-1);
        } else { // Biased rounds
            double sumBias = 0;
            for ( double b : bias )
                sumBias += b;
            double choice = session.RNG.uniform(0.0, sumBias);
            for ( size_t i = 0; i < bias.size(); i++ ) {
                choice -= bias[i];
                if ( choice < 0 ) {
                    nextStimIdx = i;
                    break;
                }
            }
        }
    } else if ( settings.randomOrder == 1 )
        nextStimIdx = session.RNG.uniform<quint32>(0, stims.size()-1);
    else
        nextStimIdx = (stimIdx + 1) % stims.size();
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

    output.error[epoch] = std::sqrt(p_err[0].err / std::ceil((stims.at(stimIdx).tObsEnd-stims.at(stimIdx).tObsBegin)/session.project.dt())); // RMSE
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
    } while ( daq->throttledFor(stims.at(stimIdx)) > 0 );

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
                if ( p[p_err[i].idx] < p.min )
                    p[p_err[i].idx] = p.min;
                if ( p[p_err[i].idx] > p.max )
                    p[p_err[i].idx] = p.max;
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
            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( iParam == stimIdx )
                p[p_err[i].idx] = session.RNG.uniform(p.min, p.max);
            else
                p[p_err[i].idx] = p[p_err[source].idx];
        }
    }

    if ( !session.daqData().cache.active )
        QThread::msleep(daq->throttledFor(stims.at(stimIdx)));
}

void GAFitter::finalise()
{
    if ( aborted )
        return;

    std::vector<std::vector<errTupel>> f_err(stims.size(), std::vector<errTupel>(lib.project.expNumCandidates()));

    // Evaluate existing population on all stims
    for ( stimIdx = 0; !aborted && stimIdx < stims.size(); stimIdx++ ) {
        // Stimulate
        stimulate(stims.at(stimIdx));

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

    if ( aborted )
        return;

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
            sumRank[ranked[i].idx].err += i;
        }
    }
    // Sort by sumRank, ascending
    std::sort(sumRank.begin(), sumRank.end(), &errTupelSort);

    for ( size_t i = 0; i < stims.size(); i++ ) {
        output.finalParams[i] = lib.adjustableParams[i][sumRank[0].idx];
        for ( size_t j = 0; j < sumRank.size(); j++ ) {
            if ( f_err[i][j].idx == sumRank[0].idx ) {
                output.finalError[i] = f_err[i][j].err;
                break;
            }
        }
    }
    output.final = true;
}

void GAFitter::stimulate(const Stimulation &I)
{
    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.VC = true;

    // Set up DAQ
    daq->VC = true;
    daq->reset();
    daq->run(I);

    // Stimulate both
    while ( lib.t < I.duration ) {
        daq->next();
        lib.Imem = daq->current;
        lib.Vmem = getCommandVoltage(I, lib.t);
        pushToQ(qT + lib.t, daq->voltage, daq->current, lib.Vmem);
        lib.getErr = (lib.t >= I.tObsBegin && lib.t < I.tObsEnd);
        lib.step();
    }

    qT += I.duration;
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
