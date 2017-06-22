#include "wavegen.h"
#include <algorithm>
#include <cassert>
#include <QDataStream>
#include <QFile>
#include "cuda_helper.h"
#include "util.h"
#include "session.h"

QString Wavegen::permute_action = QString("permute");
quint32 Wavegen::permute_magic = 0xf1d08bb4;
quint32 Wavegen::permute_version = 100;

QString Wavegen::sigmaAdjust_action = QString("sigmaAdjust");
quint32 Wavegen::sigmaAdjust_magic = 0xf084d24b;
quint32 Wavegen::sigmaAdjust_version = 101;

QString Wavegen::search_action = QString("search");
quint32 Wavegen::search_magic = 0x8a33c402;
quint32 Wavegen::search_version = 100;

Wavegen::Wavegen(Session &session) :
    SessionWorker(session),
    searchd(session.wavegenData()),
    stimd(session.stimulationData()),
    lib(session.project.wavegen()),
    RNG(),
    mapeStats(searchd.historySize, mapeArchive.end()),
    aborted(false)
{
    connect(this, SIGNAL(didAbort()), this, SLOT(clearAbort()));
}

void Wavegen::load(const QString &action, const QString &args, QFile &results)
{
    if ( action == permute_action )
        permute_load(results);
    else if ( action == sigmaAdjust_action )
        sigmaAdjust_load(results);
    else if ( action == search_action )
        search_load(results, args);
    else
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
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

void Wavegen::abort()
{
    aborted = true;
    emit didAbort();
}

void Wavegen::clearAbort()
{
    aborted = false;
}

void Wavegen::permute()
{
    if ( aborted )
        return;
    if ( !lib.project.wgPermute() ) {
        emit done();
        return;
    }

    // If the number of permuted models doesn't fit into thread blocks (very likely),
    // generate a few uncorrelated random parameter sets as padding
    int numPermutedGroups = 1;
    for ( AdjustableParam &p : lib.adjustableParams ) {
        numPermutedGroups *= p.wgPermutations + 1;
    }
    int numRandomGroups = lib.numGroups - numPermutedGroups;

    // First, generate the values:
    QVector<QVector<scalar>> allvalues(lib.adjustableParams.size());
    auto values = allvalues.begin();
    for ( AdjustableParam &p : lib.adjustableParams ) {
        values->reserve(p.wgPermutations + 1 + numRandomGroups);
        values->push_back(p.initial);
        if ( p.wgNormal ) {
            // Draw both permuted and uncorrelated random groups from normal distribution
            for ( int i = 0; i < p.wgPermutations + numRandomGroups; i++ ) {
                scalar v = p.multiplicative
                    ? (p.initial * RNG.variate<scalar, std::lognormal_distribution>(0, p.wgSD))
                    : RNG.variate<scalar, std::normal_distribution>(p.initial, p.wgSD);
                if ( v > p.max )
                    v = p.max;
                else if ( v < p.min )
                    v = p.min;
                values->push_back(v);
            }
        } else {
            // Permuted groups: Space evenly over the parameter range
            auto space = p.multiplicative ? linSpace : logSpace;
            for ( int i = 0; i < p.wgPermutations; i++ ) {
                values->push_back(space(p.min, p.max, p.wgPermutations, i));
            }
            // Random groups: Draw from uniform distribution
            if ( p.multiplicative ) {
                double min = std::log(p.min), max = std::log(p.max);
                for ( int i = 0; i < numRandomGroups; i++ )
                    values->push_back(std::exp(RNG.uniform(min, max)));
            } else {
                for ( int i = 0; i < numRandomGroups; i++ )
                    values->push_back(RNG.uniform(p.min, p.max));
            }
        }
        ++values;
    }

    permute_apply(allvalues, numPermutedGroups, numRandomGroups);

    QFile file(session.log(this, permute_action));
    permute_save(file, allvalues, numPermutedGroups, numRandomGroups);

    emit done();
}

void Wavegen::permute_apply(const QVector<QVector<scalar>> &allvalues, int numPermutedGroups, int numRandomGroups)
{
    int stride = 1;
    auto values = allvalues.begin();
    for ( AdjustableParam &p : lib.adjustableParams ) {
        // Populate this parameter, interleaving them in numGroupsPerBlock-periodic fashion
        // This algorithm is designed to maintain sanity rather than memory locality, so it hits each
        // model group in turn, skipping from warp to warp to fill out that group before moving to the next one.
        for ( int group = 0, permutation = p.wgPermutations; group < numPermutedGroups; group++ ) {
            int offset = baseModelIndex(group);
            if ( group % stride == 0)
                permutation = (permutation + 1) % (p.wgPermutations + 1);
            for ( int i = 0, end = lib.adjustableParams.size() + 1; i < end; i++ ) {
                p[i*lib.numGroupsPerBlock + offset] = values->at(permutation);
            }
        }
        for ( int randomGroup = 0; randomGroup < numRandomGroups; randomGroup++ ) {
            int offset = baseModelIndex(randomGroup+numPermutedGroups);
            for ( int i = 0, end = lib.adjustableParams.size() + 1; i < end; i++ ) {
                p[i*lib.numGroupsPerBlock + offset] = values->at(p.wgPermutations + 1 + randomGroup);
            }
        }

        // Permutation stride starts out at 1 and increases from one parameter to the next
        stride *= p.wgPermutations + 1;
        ++values;
    }
}

void Wavegen::permute_save(QFile &file, const QVector<QVector<scalar>> &values, int numPermutedGroups, int numRandomGroups)
{
    QDataStream os;
    if ( !openSaveStream(file, os, permute_magic, permute_version) )
        return;
    os << qint32(numPermutedGroups) << qint32(numRandomGroups);
    os << values;
}

void Wavegen::permute_load(QFile &file)
{
    QDataStream is;
    quint32 version = openLoadStream(file, is, permute_magic);
    if ( version < 100 || version > 100 )
        throw std::runtime_error(std::string("File version mismatch: ") + file.fileName().toStdString());
    QVector<QVector<scalar>> values;
    qint32 numPermutedGroups, numRandomGroups;
    is >> numPermutedGroups >> numRandomGroups;
    is >> values;
    permute_apply(values, numPermutedGroups, numRandomGroups);
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
    Stimulation I;
    I.duration = session.runData().settleDuration;
    I.baseV = stimd.baseV;
    I.clear();
    for ( int group = 0; group < lib.numGroups; group++ )
        lib.waveforms[group] = I;
    lib.pushWaveforms();
    lib.getErr = false;
    lib.settling = true;
    lib.push();
    lib.step();
}

void Wavegen::adjustSigmas()
{
    if ( aborted )
        return;
    detune();
    settle();
    for ( int i = 0; i < lib.numModels; i++ )
        lib.err[i] = 0;
    lib.getErr = true;
    lib.targetParam = -1;

    // Generate a set of random waveforms,
    // simulate each (in turn across all model permutations, or in parallel), and collect the
    // per-parameter average deviation from the base model produced by that parameter's detuning.
    std::vector<double> sumParamErr(lib.adjustableParams.size(), 0);
    std::vector<Stimulation> waves;
    int end = lib.project.wgPermute()
            ? searchd.numSigmaAdjustWaveforms
              // round numSigAdjWaves up to nearest multiple of nGroups to fully occupy each iteration:
            : ((searchd.numSigmaAdjustWaveforms + lib.numGroups - 1) / lib.numGroups);
    for ( int i = 0; i < end && !aborted; i++ ) {
        lib.pushErr();

        // Generate random wave/s
        if ( lib.project.wgPermute() ) {
            if ( !i )
                waves.resize(1);
            waves[0] = getRandomStim();
        } else {
            if ( !i )
                waves.resize(lib.numGroups);
            for ( Stimulation &w : waves )
                w = getRandomStim();
        }

        // Simulate
        stimulate(waves);

        // Collect per-parameter error
        lib.pullErr();
        for ( int j = 0; j < lib.numModels; j++ ) {
            int param = (j % lib.numModelsPerBlock) / lib.numGroupsPerBlock;
            if ( param && !isnan(lib.err[j]) ) // Collect error for stable detuned models only
                sumParamErr[param-1] += lib.err[j];
            lib.err[j] = 0;
        }
    }

    std::vector<double> meanParamErr(sumParamErr);
    for ( double & e : meanParamErr ) {
        e /= end * lib.numGroups * stimd.duration/lib.project.dt() * lib.simCycles;
    }

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

    // Report
    std::cout << "Perturbation adjustment complete." << std::endl;
    std::cout << "Mean deviation in nA across all random waveforms, adjustment, and new perturbation factor:" << std::endl;
    for ( int i = 0; i < (int)lib.adjustableParams.size(); i++ )
        std::cout << lib.adjustableParams[i].name << ":\t" << meanParamErr[i] << '\t'
                  << sigmaAdjust[i] << '\t' << lib.adjustableParams[i].adjustedSigma << std::endl;
    std::cout << "These adjustments are applied to all future actions." << std::endl;

    QFile file(session.log(this, sigmaAdjust_action));
    sigmaAdjust_save(file);

    emit done();
}

void Wavegen::sigmaAdjust_save(QFile &file)
{
    QDataStream os;
    if ( !openSaveStream(file, os, sigmaAdjust_magic, sigmaAdjust_version) )
        return;
    for ( const AdjustableParam &p : lib.adjustableParams )
        os << p.adjustedSigma;
}

void Wavegen::sigmaAdjust_load(QFile &file)
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
}

void Wavegen::stimulate(const std::vector<Stimulation> &stim)
{
    lib.settling = false;
    if ( lib.project.wgPermute() ) {
        const Stimulation &s = stim.at(0);
        for ( int group = 0; group < lib.numGroups; group++ )
            lib.waveforms[group] = s;
    } else { //-------------- !m.cfg.permute ------------------------------------------------------------
        assert((int)stim.size() >= lib.numGroups);
        for ( int group = 0; group < lib.numGroups; group++ ) {
            lib.waveforms[group] = stim[group];
        }
    }
    lib.pushWaveforms();
    lib.step();
}
