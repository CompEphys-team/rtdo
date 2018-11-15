#include "wavegen.h"
#include "session.h"
#include <functional>

QString Wavegen::ee_action = QString("ee_search");
quint32 Wavegen::ee_magic = 0xc9fd545f;
quint32 Wavegen::ee_version = 100;

void Wavegen::elementaryEffects()
{
    session.queue(actorName(), ee_action, "", new Result());
}

void prepareModels(Session &session, UniversalLibrary &ulib, const WavegenData &searchd)
{
    int nParams = ulib.adjustableParams.size(), nextParam = 0;
    bool initial = true;
    std::vector<scalar> values(nParams);
    size_t nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        if ( i % searchd.trajectoryLength == 0 ) { // Pick a starting point
            // For the benefit of rerandomiseParameters && useBaseParameters, provide each new stim with a trajectory from base model
            if ( i % nModelsPerStim == 0 )
                initial = true;
            if ( searchd.useBaseParameters && initial ) // Reset to base model
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j].initial;
            else if ( searchd.rerandomiseParameters || i < nModelsPerStim ) // Randomise uniformly
                for ( int j = 0; j < nParams; j++ )
                    values[j] = session.RNG.uniform(ulib.adjustableParams[j].min, ulib.adjustableParams[j].max);
            else // Copy from first stim
                for ( int j = 0; j < nParams; j++ )
                    values[j] = ulib.adjustableParams[j][i % nModelsPerStim];
        } else {
            // Add a sigma-sized step to one parameter at a time
            values[nextParam] += ulib.adjustableParams[nextParam].sigma;
            nextParam = (nextParam+1) % nParams;
            if ( nextParam == 0 )
                initial = false;
        }

        for ( size_t j = 0; j < values.size(); j++ )
            ulib.adjustableParams[j][i] = values[j];
    }
}

void settleModels(const Session &session, UniversalLibrary &ulib)
{
    const RunData &rd = session.runData();

    ulib.setSingularRund();
    ulib.clampGain[0] = rd.clampGain;
    ulib.accessResistance[0] = rd.accessResistance;
    ulib.iSettleDuration[0] = rd.settleDuration / rd.dt;
    ulib.Imax[0] = rd.Imax;
    ulib.dt[0] = rd.dt;

    ulib.setSingularTarget();
    ulib.targetOffset[0] = 0;

    ulib.setSingularStim();
    ulib.stim[0].baseV = session.stimulationData().baseV;

    ulib.assignment = ulib.assignment_base | ASSIGNMENT_SETTLE_ONLY | ASSIGNMENT_MAINTAIN_STATE;
    ulib.push();
    ulib.run();
}

void pushStimsAndObserve(const std::vector<iStimulation>&stims, UniversalLibrary &ulib, int nModelsPerStim, int blankCycles)
{
    ulib.setSingularStim(false);
    for ( size_t i = 0; i < ulib.NMODELS; i++ ) {
        ulib.stim[i] = stims[i / nModelsPerStim];
    }
    ulib.push(ulib.stim);
    ulib.observe_no_steps(blankCycles);
}

QVector<double> getDeltabar(Session &session, UniversalLibrary &ulib)
{
    const WavegenData &searchd = session.wavegenData();
    int nModelsPerStim = searchd.nTrajectories * searchd.trajectoryLength;
    std::vector<iStimulation> stims(ulib.NMODELS / nModelsPerStim);
    for ( iStimulation &stim : stims )
        stim = session.wavegen().getRandomStim();

    ulib.iSettleDuration[0] = 0;
    ulib.push(ulib.iSettleDuration);

    pushStimsAndObserve(stims, ulib, nModelsPerStim, session.gaFitterSettings().cluster_blank_after_step / session.runData().dt);

    ulib.assignment = ulib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_PREVTHREAD;
    ulib.run();

    std::vector<double> dbar = ulib.find_deltabar(searchd.trajectoryLength, searchd.nTrajectories, stims[0].duration);
    return QVector<double>::fromStdVector(dbar);
}

bool Wavegen::ee_exec(QFile &file, Result *result)
{
    current = Archive(-1, searchd, *result);
    delete result;
    result = nullptr;

    emit startedSearch(-1);

    UniversalLibrary &ulib = session.project.universal();
    const int nModelsPerStim = searchd.trajectoryLength * searchd.nTrajectories;
    const int nStimsPerEpoch = ulib.NMODELS / nModelsPerStim;

    ulib.resizeOutput(istimd.iDuration);
    prepareModels(session, ulib, searchd);
    settleModels(session, ulib);

    // Use the settling period to generate the first two sets of stims
    std::vector<iStimulation> waves_ep1(nStimsPerEpoch), waves_ep2(nStimsPerEpoch);
    for ( iStimulation &w : waves_ep1 )
        w = getRandomStim();
    for ( iStimulation &w : waves_ep2 )
        w = getRandomStim();
    size_t nInitialStims = 2*nStimsPerEpoch;
    std::vector<iStimulation> *returnedWaves = &waves_ep1, *newWaves = &waves_ep2;

    // Run a set of stims through a deltabar finding mission
    current.deltabar = getDeltabar(session, ulib);


    return false;
}

void Wavegen::ee_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, ee_magic, ee_version) )
        return;
    Archive &arch = m_archives.back();
    os << quint32(arch.precision);
    os << quint32(arch.iterations);
    os << arch.nCandidates << arch.nInsertions << arch.nReplacements << arch.nElites;
    os << arch.deltabar;

    os << quint32(arch.elites.size());
    os << quint32(arch.elites.front().bin.size());

    // Separate waves into unique and shared to reduce the number of lookups
    std::vector<iStimulation*> w_unique, w_shared;
    for ( MAPElite const& e : arch.elites ) {
        os << e.fitness;
        for ( size_t b : e.bin )
            os << quint32(b);

        if ( e.wave.unique() ) {
            w_unique.push_back(e.wave.get());
            os << qint32(-w_unique.size());
        } else {
            size_t i;
            for ( i = 0; i < w_shared.size(); i++ )
                if ( e.wave.get() == w_shared[i] )
                    break;
            if ( i == w_shared.size() )
                w_shared.push_back(e.wave.get());
            os << qint32(i);
        }
    }
    os << quint32(w_unique.size());
    for ( const iStimulation *pstim : w_unique )
        os << *pstim;
    os << quint32(w_shared.size());
    for ( const iStimulation *pstim : w_shared )
        os << *pstim;
}

void Wavegen::ee_load(QFile &file, const QString &, Result r)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, ee_magic);
    if ( version < 100 || version > ee_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());

    m_archives.emplace_back(-1, searchd, r);
    Archive &arch = m_archives.back();

    quint32 precision, iterations, archSize, nBins;
    is >> precision >> iterations;
    arch.precision = precision;
    arch.iterations = iterations;
    is >> arch.nCandidates >> arch.nInsertions >> arch.nReplacements >> arch.nElites;
    is >> arch.deltabar;

    is >> archSize >> nBins;
    arch.elites.resize(archSize);
    std::vector<qint32> stimIdx(archSize);
    auto idxIt = stimIdx.begin();
    quint32 tmp;
    for ( auto el = arch.elites.begin(); el != arch.elites.end(); el++, idxIt++ ) {
        is >> el->fitness;
        el->bin.resize(nBins);
        for ( size_t &b : el->bin ) {
            is >> tmp;
            b = size_t(tmp);
        }
        is >> *idxIt;
    }

    quint32 uniqueSize, sharedSize;
    is >> uniqueSize;
    std::vector<std::shared_ptr<iStimulation>> w_unique(uniqueSize);
    for ( std::shared_ptr<iStimulation> &ptr : w_unique ) {
        ptr.reset(new iStimulation);
        is >> *ptr;
    }
    is >> sharedSize;
    std::vector<std::shared_ptr<iStimulation>> w_shared(sharedSize);
    for ( std::shared_ptr<iStimulation> &ptr : w_shared ) {
        ptr.reset(new iStimulation);
        is >> *ptr;
    }

    idxIt = stimIdx.begin();
    for ( auto el = arch.elites.begin(); el != arch.elites.end(); el++, idxIt++ ) {
        if ( *idxIt < 0 )
            el->wave = w_unique[-1 - *idxIt];
        else
            el->wave = w_shared[*idxIt];
    }
}
