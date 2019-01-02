#include "wavegen.h"
#include <algorithm>
#include <cassert>
#include <QDataStream>
#include <QFile>
#include "cuda_helper.h"
#include "util.h"
#include "session.h"

Wavegen::Wavegen(Session &session) :
    SessionWorker(session),
    searchd(session.wavegenData()),
    stimd(session.stimulationData()),
    lib(session.project.wavegen()),
    ulib(session.project.universal())
{
}

Result *Wavegen::load(const QString &action, const QString &args, QFile &results, Result r)
{
    if ( action == cluster_action )
        return cluster_load(results, args, r);
    else if ( action == bubble_action )
        return bubble_load(results, args, r);
    else
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
}

bool Wavegen::execute(QString action, QString, Result *res, QFile &file)
{
    clearAbort();
    istimd = iStimData(stimd, session.runData().dt);
    if ( action == cluster_action )
        return cluster_exec(file, res);
    else if ( action == bubble_action )
        return bubble_exec(file, res);
    else
        return false;
}

void Wavegen::initModels(bool withBase)
{
    withBase &= searchd.useBaseParameters;
    scalar value;
    for ( AdjustableParam &p : lib.adjustableParams ) {
        for ( int block = 0; block < lib.numBlocks; block++ ) {
            for ( int group = 0; group < lib.numGroupsPerBlock; group++ ) {
                if ( withBase && (lib.numGroupsPerBlock*block + group) % searchd.nGroupsPerWave == 0 )
                    value = p.initial;
                else
                    value = session.RNG.uniform<scalar>(p.min, p.max);
                for ( size_t model = 0; model < lib.adjustableParams.size()+1; model++ ) {
                    p[group + lib.numGroupsPerBlock*model + lib.numModelsPerBlock*block] = value;
                }
            }
        }
    }
}

void Wavegen::detune()
{
    int k = 0;
    for ( AdjustableParam &p : lib.adjustableParams ) {
        scalar sigma = p.adjustedSigma + (p.multiplicative ? 1 : 0);
        for ( int group = 0, paramOffset = ++k * lib.numGroupsPerBlock; group < lib.numGroups; group++ ) {
            int tuned             = baseModelIndex(group),  // The index of the tuned/base model
                detune            = tuned + paramOffset;    // The index of the model being detuned
            scalar newp;
            if ( p.multiplicative ) {
                newp = p[tuned] * sigma; // Get original value from base model & detune
                if ( newp > p.max || newp < p.min )
                    newp = p[tuned] * (2-sigma); // Mirror multiplicative sigma around 1 (up is down)
            } else {
                newp = p[tuned] + sigma;
                if ( newp > p.max || newp < p.min )
                    newp = p[tuned] - sigma;
            }
            p[detune] = newp; // Place shifted value into detuned model
        }
    }
}

void Wavegen::settle()
{
    // Simulate for a given time, retaining the final state
    iStimulation I;
    I.duration = lrint(session.runData().settleDuration / session.runData().dt);
    I.baseV = stimd.baseV;
    I.clear();
    pushStims({I});
    lib.getErr = false;
    lib.settling = true;
    lib.dt = session.runData().dt;
    lib.simCycles = 1;
    lib.push();
    lib.step();
    lib.settling = false;
}

std::vector<double> Wavegen::getMeanParamError()
{
    for ( int i = 0; i < lib.numModels; i++ )
        lib.err[i] = 0;
    lib.getErr = true;
    lib.targetParam = -1;

    // Generate a set of random waveforms,
    // simulate each (in turn across all model permutations, or in parallel), and collect the
    // average deviation from the base model produced by that parameter's detuning.
    std::vector<double> sumParamErr(lib.adjustableParams.size(), 0);
    std::vector<size_t> nValid(lib.adjustableParams.size(), 0);
    std::vector<iStimulation> waves(lib.numGroups);
    // Round numSigAdjWaves up to nearest multiple of nGroups to fully occupy each iteration:
    int end = (searchd.numSigmaAdjustWaveforms + lib.numGroups - 1) / lib.numGroups;
    for ( int i = 0; i < end && !isAborted(); i++ ) {
        lib.pushErr();

        // Generate random wave/s
        for ( iStimulation &w : waves )
            w = getRandomStim(stimd, istimd);

        // Simulate
        pushStims(waves);
        lib.step();

        // Collect per-parameter error
        lib.pullErr();
        for ( int j = 0; j < lib.numModels; j++ ) {
            int param = (j % lib.numModelsPerBlock) / lib.numGroupsPerBlock; // 1-based (param==0 is base model)
            if ( param && !isnan(lib.err[j]) ) { // Collect error for stable detuned models only
                sumParamErr[param-1] += lib.err[j];
                nValid[param-1]++;
            }
            lib.err[j] = 0;
        }
    }
    lib.pushErr();

    for ( size_t i = 0; i < sumParamErr.size(); i++ )
        sumParamErr[i] /= nValid[i] * istimd.iDuration;

    return sumParamErr;
}

void Wavegen::pushStims(const std::vector<iStimulation> &stim)
{
    if ( stim.size() == size_t(lib.numGroups) )
        for ( int group = 0; group < lib.numGroups; group++ )
            lib.waveforms[group] = stim[group];
    else if ( stim.size() == lib.numGroups / searchd.nGroupsPerWave )
        for ( int group = 0; group < lib.numGroups; group++ )
            lib.waveforms[group] = stim[group / searchd.nGroupsPerWave];
    else if ( stim.size() == 1 )
        for ( int group = 0; group < lib.numGroups; group++ )
            lib.waveforms[group] = stim[0];
    else
        throw std::runtime_error("Invalid number of stimulations.");

    lib.pushWaveforms();
}

void Wavegen::diagnose(iStimulation I, double dt, int simCycles)
{
    initModels();
    detune();
    settle();
    lib.dt = dt;
    lib.simCycles = simCycles;
    lib.diagnose(I);
    lib.dt = session.runData().dt;
    lib.simCycles = 1;
}
