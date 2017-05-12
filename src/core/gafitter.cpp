#include "gafitter.h"
#include "session.h"
#include "supportcode.h"

const QString GAFitter::action = QString("fit");
const quint32 GAFitter::magic = 0xadb8d269;
const quint32 GAFitter::version = 100;

GAFitter::GAFitter(Session &session, DAQ *daq) :
    SessionWorker(session),
    lib(session.project.experiment()),
    settings(session.gaFitterSettings()),
    simulator(lib.createSimulator()),
    daq(daq ? daq : simulator),
    RNG(),
    aborted(false),
    deck(session),
    p_err(lib.project.expNumCandidates()),
    output(*this)
{
    connect(this, &GAFitter::didAbort, this, &GAFitter::clearAbort);
}

GAFitter::~GAFitter()
{
    lib.destroySimulator(simulator);
}

void GAFitter::stageDeck(WaveSource deck)
{
    if ( deck.type != WaveSource::Deck )
        throw std::runtime_error("Wave source for GAFitter must be a deck.");
    stagedDeck = std::move(deck);
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

GAFitter::Output::Output(const GAFitter &f) :
    params(f.settings.maxEpochs, std::vector<scalar>(f.lib.adjustableParams.size())),
    error(f.settings.maxEpochs),
    stimIdx(f.settings.maxEpochs),
    epochs(0)
{
    deck.session =& f.session;
}

void GAFitter::run()
{
    if ( aborted )
        return;

    // Prepare
    output = Output(*this);
    output.deck = stagedDeck;
    deck = *output.deck.deck();
    stimIdx = 0;

    Stimulation hold;
    hold.duration = session.runData().settleDuration;
    hold.clear();

    // Fit
    populate();
    for ( epoch = 0; !aborted && !finished(); epoch++ ) {
        const Stimulation &stim = deck.stimulations().at(stimIdx);

        // Settle
        hold.baseV = stim.baseV;
        settle(hold);

        // Stimulate
        stimulate(stim);

        // Advance
        lib.pullErr();
        procreate();
        lib.push();

        if ( settings.randomOrder )
            stimIdx = RNG.uniform<quint32>(0, deck.stimulations().size()-1);
        else
            stimIdx = (stimIdx+1) % deck.stimulations().size();

        emit progress(epoch);
    }

    // Finish
    output.epochs = epoch;
    m_results.push_back(output);
    emit done();

    // Save
    QFile file(session.log(this, action));
    QDataStream os;
    if ( openSaveStream(file, os, magic, version) ) {
        const Output &out = m_results.back();
        os << out.deck << out.epochs;
        for ( quint32 i = 0; i < out.epochs; i++ ) {
            os << out.stimIdx[i] << out.error[i];
            for ( const scalar &p : out.params[i] )
                os << p;
        }
    }
}

void GAFitter::load(const QString &act, const QString &, QFile &results)
{
    if ( act != action )
        throw std::runtime_error(std::string("Unknown action: ") + act.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    Output out(*this);
    is >> out.deck >> out.epochs;
    out.stimIdx.resize(out.epochs);
    out.params.resize(out.epochs);
    out.error.resize(out.epochs);
    for ( quint32 i = 0; i < out.epochs; i++ ) {
        is >> out.stimIdx[i] >> out.error[i];
        for ( scalar &p : out.params[i] )
            is >> p;
    }
    m_results.push_back(std::move(out));
}

bool GAFitter::finished()
{
    return epoch >= settings.maxEpochs;
}

void GAFitter::populate()
{
    for ( AdjustableParam &p : lib.adjustableParams ) {
        for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
            p[i] = RNG.uniform<scalar>(p.min, p.max);
        }
    }
    for ( size_t i = 0; i < lib.project.expNumCandidates(); i++ ) {
        lib.err[i] = 0;
    }
    lib.push();
}

void GAFitter::procreate()
{
    for ( size_t i = 0; i < p_err.size(); i++ ) {
        p_err[i].idx = i;
        p_err[i].err = lib.err[i];
        lib.err[i] = 0;
    }
    std::sort(p_err.begin(), p_err.end(), [](const errTupel &x, const errTupel &y) -> bool {
        // make sure NaN is largest
        if ( std::isnan(x.err) ) return false;
        if ( std::isnan(y.err) ) return true;
        return x.err < y.err;
    });

    scalar sigma = lib.adjustableParams[stimIdx].sigma;
    if ( settings.decaySigma )
        sigma = settings.sigmaInitial * sigma * std::exp2(-double(epoch)/settings.sigmaHalflife);

    // Mutate
    for ( size_t i = p_err.size()-settings.nReinit-1; i >= settings.nElite; i-- ) {
        // Bias reproductions towards the elite in roughly linear fashion by restricting the choice range
        size_t targetSource = RNG.uniform<size_t>(0, i-settings.nElite);
        size_t otherSource = targetSource;
        if ( settings.crossover > 0 && RNG.uniform<double>(0,1) < settings.crossover )
            otherSource = RNG.uniform<size_t>(0, i-settings.nElite);
        for ( size_t iParam = 0; iParam < lib.adjustableParams.size(); iParam++ ) {
            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( iParam == stimIdx ) {
                // Mutate target param
                p[p_err[i].idx] = p.multiplicative ?
                            (p[p_err[targetSource].idx] * RNG.variate<scalar, std::lognormal_distribution>(0, sigma)) :
                            RNG.variate<scalar, std::normal_distribution>(p[p_err[targetSource].idx], sigma);
                if ( p[p_err[i].idx] < p.min )
                    p[p_err[i].idx] = p.min;
                if ( p[p_err[i].idx] > p.max )
                    p[p_err[i].idx] = p.max;
            } else {
                // Copy non-target params
                p[p_err[i].idx] = p[p_err[otherSource].idx];
            }
        }
    }

    // Reinit
    for ( size_t source = 0; source < settings.nReinit; source++ ) {
        size_t i = p_err.size() - source - 1;
        for ( size_t iParam = 0; iParam < lib.adjustableParams.size(); iParam++ ) {
            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( iParam == stimIdx )
                p[p_err[i].idx] = RNG.uniform(p.min, p.max);
            else
                p[p_err[i].idx] = p[p_err[source].idx];
        }
    }

    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.params[epoch][i] = lib.adjustableParams[i][p_err[0].idx];
    }
    output.error[epoch] = p_err[0].err;
    output.stimIdx[epoch] = stimIdx;
}

void GAFitter::stimulate(const Stimulation &I)
{
    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.VC = true;

    // Set up DAQ
    daq->reset();
    daq->run(I);

    // Stimulate both
    while ( lib.t < I.duration ) {
        daq->next();
        lib.Imem = daq->current;
        lib.Vmem = getCommandVoltage(I, lib.t);
        lib.getErr = (lib.t > I.tObsBegin && lib.t < I.tObsEnd);
        lib.step();
    }

    daq->reset();
}

void GAFitter::settle(const Stimulation &I)
{
    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.getErr = false;
    lib.VC = true;
    lib.Vmem = I.baseV;

    // Set up DAQ
    daq->reset();
    daq->run(I);

    // Stimulate both
    while ( lib.t < I.duration ) {
        daq->next();
        lib.step();
    }

    daq->reset();
}
