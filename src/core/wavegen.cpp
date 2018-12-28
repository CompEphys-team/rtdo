#include "wavegen.h"
#include <algorithm>
#include <cassert>
#include <QDataStream>
#include <QFile>
#include "cuda_helper.h"
#include "util.h"
#include "session.h"

QString Wavegen::sigmaAdjust_action = QString("sigmaAdjust");
quint32 Wavegen::sigmaAdjust_magic = 0xf084d24b;
quint32 Wavegen::sigmaAdjust_version = 101;

QString Wavegen::search_action = QString("search");
quint32 Wavegen::search_magic = 0x8a33c402;
quint32 Wavegen::search_version = 112;

Wavegen::Wavegen(Session &session) :
    SessionWorker(session),
    searchd(session.wavegenData()),
    stimd(session.stimulationData()),
    lib(session.project.wavegen())
{
}

void Wavegen::load(const QString &action, const QString &args, QFile &results, Result r)
{
    if ( action == sigmaAdjust_action )
        sigmaAdjust_load(results, r);
    else if ( action == search_action )
        search_load(results, args, r);
    else if ( action == ee_action )
        ee_load(results, args, r);
    else
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
}

bool Wavegen::execute(QString action, QString, Result *res, QFile &file)
{
    clearAbort();
    istimd = iStimData(stimd, session.runData().dt);
    if ( action == sigmaAdjust_action )
        return sigmaAdjust_exec(file, res);
    else if ( action == search_action )
        return search_exec(file, res);
    else if ( action == ee_action )
        return ee_exec(file, res);
    else
        return false;
}

std::vector<double> Wavegen::getSigmaMaxima()
{
    // Sigmas probably shouldn't exceed 10% of a parameter's range, so let's use that as a maximum:
    constexpr double factor = 0.1;
    std::vector<double> sigmax(lib.adjustableParams.size());
    int k = 0;
    for ( const AdjustableParam &p : lib.adjustableParams ) {
        if ( p.multiplicative ) {
            // Multiplicative with a range crossing or ending at 0 is daft, but this isn't the place to fail:
            if ( p.max == 0 || p.min == 0 || p.min * p.max < 0 )
                sigmax[k] = factor;
            else
                sigmax[k] = (p.max / p.min) * factor;
        } else {
            sigmax[k] = (p.max - p.min) * factor;
        }
        if ( p.sigma > sigmax[k] )
            sigmax[k] = p.sigma;
        ++k;
    }
    return sigmax;
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

void Wavegen::adjustSigmas()
{
    session.queue(actorName(), sigmaAdjust_action, "", new Result());
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

bool Wavegen::sigmaAdjust_exec(QFile &file, Result *dummy)
{
    delete dummy;
    dummy = nullptr;

    initModels(false);
    detune();
    settle();

    std::vector<double> meanParamErr = getMeanParamError();

    // Find the median of mean parameter errors:
    double medianErr; {
        std::vector<double> sortedErrs(meanParamErr);
        std::sort(sortedErrs.begin(), sortedErrs.end());
        if ( lib.adjustableParams.size() % 2 )
            medianErr = 0.5 * (sortedErrs[lib.adjustableParams.size()/2] + sortedErrs[lib.adjustableParams.size()/2 - 1]);
        else
            medianErr = sortedErrs[lib.adjustableParams.size()/2];
    }

    // Set sigmaAdjust to draw each parameter error average towards the median of parameter error averages
    // Assume a more-or-less linear relationship, where doubling the sigma roughly doubles the deviation.
    // This is a simplification, but it should work well enough for small sigmas and small increments thereof.
    std::vector<double> sigmaAdjust(lib.adjustableParams.size(), 1);
    std::vector<double> sigmax = getSigmaMaxima();
    double maxExcess = 1;
    for ( size_t k = 0; k < lib.adjustableParams.size(); k++ ) {
        sigmaAdjust[k] *= medianErr / meanParamErr[k];
        if ( sigmaAdjust[k] * lib.adjustableParams[k].adjustedSigma > sigmax[k] ) {
            double excess = sigmaAdjust[k] * lib.adjustableParams[k].adjustedSigma / sigmax[k];
            if ( excess > maxExcess )
                maxExcess = excess;
        }
    }
    // Ensure that no sigmaAdjust exceeds its sigmax boundaries:
    if ( maxExcess > 1 )
        for ( double &adj : sigmaAdjust )
            adj /= maxExcess;

    // Apply
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
        lib.adjustableParams[i].adjustedSigma = lib.adjustableParams[i].adjustedSigma*sigmaAdjust[i];
    propagateAdjustedSigma();

    // Report
    std::cout << "Perturbation adjustment complete." << std::endl;
    std::cout << "Mean deviation in nA across all random waveforms, adjustment, and new perturbation factor:" << std::endl;
    for ( int i = 0; i < (int)lib.adjustableParams.size(); i++ )
        std::cout << lib.adjustableParams[i].name << ":\t" << meanParamErr[i] << '\t'
                  << sigmaAdjust[i] << '\t' << lib.adjustableParams[i].adjustedSigma << std::endl;
    std::cout << "These adjustments are applied to all future actions." << std::endl;

    // Save
    sigmaAdjust_save(file);

    emit done();
    return true;
}

void Wavegen::sigmaAdjust_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, sigmaAdjust_magic, sigmaAdjust_version) )
        return;
    for ( const AdjustableParam &p : lib.adjustableParams )
        os << p.adjustedSigma;
}

void Wavegen::sigmaAdjust_load(QFile &file, Result)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, sigmaAdjust_magic);
    if ( version < 100 || version > sigmaAdjust_version )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());
    if ( version < 101 ) {
        QVector<double> sigmaAdjust;
        is >> sigmaAdjust;
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
            lib.adjustableParams[i].adjustedSigma = lib.adjustableParams[i].sigma * sigmaAdjust[i];
    } else {
        for ( AdjustableParam &p : lib.adjustableParams )
            is >> p.adjustedSigma;
    }
    propagateAdjustedSigma();
}

void Wavegen::propagateAdjustedSigma()
{
    UniversalLibrary &uni(session.project.universal());
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        uni.adjustableParams[i].adjustedSigma = lib.adjustableParams[i].adjustedSigma;
    }
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
