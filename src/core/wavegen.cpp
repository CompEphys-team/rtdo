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
quint32 Wavegen::sigmaAdjust_version = 100;

QString Wavegen::search_action = QString("search");
quint32 Wavegen::search_magic = 0x8a33c402;
quint32 Wavegen::search_version = 100;

Wavegen::Wavegen(Session &session) :
    session(session),
    searchd(session.wavegenData()),
    stimd(session.stimulationData()),
    lib(session.project.wavegen()),
    RNG(),
    sigmaAdjust(lib.adjustableParams.size(), 1.0),
    sigmax(getSigmaMaxima()),
    mapeStats(searchd.historySize, mapeArchive.end()),
    aborted(false),
    completedArchives(lib.adjustableParams.size()),
    archivePrecision(lib.adjustableParams.size())
{
    connect(this, SIGNAL(didAbort()), this, SLOT(clearAbort()));
}

void Wavegen::load(const QString &action, const QString &args, const QString &results)
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
                    ? RNG.variate<scalar, std::lognormal_distribution>(p.initial, p.wgSD)
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

    QString filename = session.log(this, permute_action);
    permute_save(filename, allvalues, numPermutedGroups, numRandomGroups);

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
    settled.clear();
}

void Wavegen::permute_save(const QString &filename, const QVector<QVector<scalar>> &values, int numPermutedGroups, int numRandomGroups)
{
    QDataStream os;
    if ( !Session::openSaveStream(filename, os, permute_magic, permute_version) )
        return;
    os << qint32(numPermutedGroups) << qint32(numRandomGroups);
    os << values;
}

void Wavegen::permute_load(const QString &filename)
{
    QDataStream is;
    quint32 version = Session::openLoadStream(filename, is, permute_magic);
    if ( version < 100 || version > 100 )
        throw std::runtime_error(std::string("File version mismatch: ") + filename.toStdString());
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
        scalar sigma = p.sigma * sigmaAdjust[k] + (p.multiplicative ? 1 : 0);
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
    // Simulate for a given time
    Stimulation I;
    I.duration = searchd.settleTime;
    I.baseV = stimd.baseV;
    I.clear();
    for ( int group = 0; group < lib.numGroups; group++ )
        lib.waveforms[group] = I;
    lib.pushWaveforms();
    lib.getErr = false;
    lib.t = 0;
    lib.iT = 0;
    lib.push();
    while ( lib.t < searchd.settleTime ) {
        lib.step();
    }
    lib.pull();
    if ( lib.project.wgPermute() ) {
        // Collect the state variables of every base model, i.e. the tuned version
        settled = std::list<std::vector<scalar>>(lib.stateVariables.size(), std::vector<scalar>(lib.numGroups));
        auto iter = settled.begin();
        for ( StateVariable &v : lib.stateVariables ) {
            for ( int group = 0; group < lib.numGroups; group++ ) {
                (*iter)[group] = v[baseModelIndex(group)];
            }
            ++iter;
        }

        // Provide some auditing info
        std::vector<scalar> V;
        int valid = 0;
        for ( int group = 0; group < lib.numGroups; group++ ) {
            scalar const& v = lib.stateVariables[0].v[baseModelIndex(group)];
            V.push_back(v);
            if ( v > stimd.baseV-5 && v < stimd.baseV+5 )
                valid++;
        }
        std::sort(V.begin(), V.end());
        std::cout << "Settled all permuted models to holding potential of " << stimd.baseV << " mV for "
                  << searchd.settleTime << " ms." << std::endl;
        std::cout << "Median achieved potential: " << V[lib.numGroups/2] << " mV (95% within [" << V[lib.numGroups/20]
                  << " mV, " << V[lib.numGroups/20*19] << " mV]), " << valid << "/" << lib.numGroups
                  << " models within +-5 mV of holding." << std::endl;
    } else {
        // Collect the state of one base model
        settled = std::list<std::vector<scalar>>(lib.stateVariables.size(), std::vector<scalar>(1));
        auto iter = settled.begin();
        for ( StateVariable &v : lib.stateVariables ) {
            (*iter++)[0] = v[0];
        }

        std::cout << "Settled base model to holding potential of " << stimd.baseV << " mV for " << searchd.settleTime << " ms, "
                  << "achieved " << (*settled.begin())[0] << " mV." << std::endl;
    }
}

void Wavegen::restoreSettled()
{
    if ( settled.empty() ) {
        settle();
        return;
    }

    // Restore to previously found settled state
    auto iter = settled.begin();
    if ( lib.project.wgPermute() ) {
        for ( StateVariable &v : lib.stateVariables ) {
            for ( int i = 0; i < lib.numModels; i++ ) {
                int group = i % lib.numGroupsPerBlock            // Group index within the block
                        + (i/lib.numModelsPerBlock) * lib.numGroupsPerBlock; // Offset (in group space) of the block this model belongs to
                v[i] = (*iter)[group];
            }
            ++iter;
        }
    } else {
        for ( StateVariable &v : lib.stateVariables ) {
            for ( int i = 0; i < lib.numModels; i++ ) {
                v[i] = (*iter)[0];
            }
            ++iter;
        }
    }
    lib.push();
}

void Wavegen::adjustSigmas()
{
    if ( aborted )
        return;
    restoreSettled();
    detune();
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
        restoreSettled();
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
    double maxExcess = 1;
    for ( size_t k = 0; k < lib.adjustableParams.size(); k++ ) {
        sigmaAdjust[k] *= medianErr / meanParamErr[k];
        if ( sigmaAdjust[k] * lib.adjustableParams[k].sigma > sigmax[k] ) {
            double excess = sigmaAdjust[k] * lib.adjustableParams[k].sigma / sigmax[k];
            if ( excess > maxExcess )
                maxExcess = excess;
        }
    }
    // Ensure that no sigmaAdjust exceeds its sigmax boundaries:
    if ( maxExcess > 1 )
        for ( double &adj : sigmaAdjust )
            adj /= maxExcess;

    std::cout << "Perturbation adjustment complete." << std::endl;
    std::cout << "Mean deviation in nA across all random waveforms, adjustment, and new perturbation factor:" << std::endl;
    for ( int i = 0; i < (int)lib.adjustableParams.size(); i++ )
        std::cout << lib.adjustableParams[i].name << ":\t" << meanParamErr[i] << '\t'
                  << sigmaAdjust[i] << '\t' << lib.adjustableParams[i].sigma*sigmaAdjust[i] << std::endl;

    QString filename = session.log(this, sigmaAdjust_action);
    sigmaAdjust_save(filename);

    emit done();
}

void Wavegen::sigmaAdjust_save(const QString &filename)
{
    QDataStream os;
    if ( !Session::openSaveStream(filename, os, sigmaAdjust_magic, sigmaAdjust_version) )
        return;
    os << sigmaAdjust;
}

void Wavegen::sigmaAdjust_load(const QString &filename)
{
    QDataStream is;
    quint32 version = Session::openLoadStream(filename, is, sigmaAdjust_magic);
    if ( version < 100 || version > 100 )
        throw std::runtime_error(std::string("File version mismatch: ") + filename.toStdString());
    is >> sigmaAdjust;
}

void Wavegen::stimulate(const std::vector<Stimulation> &stim)
{
    lib.t = 0;
    lib.iT = 0;
    if ( lib.project.wgPermute() ) {
        const Stimulation &s = stim.at(0);
        for ( int group = 0; group < lib.numGroups; group++ )
            lib.waveforms[group] = s;
        lib.pushWaveforms();
        while ( lib.t < s.duration ) {
            lib.final = lib.t + lib.project.dt() >= s.duration;
            lib.step();
        }
    } else { //-------------- !m.cfg.permute ------------------------------------------------------------
        assert((int)stim.size() >= lib.numGroups);
        double maxDuration = 0.0, minDuration = stim[0].duration;
        for ( int group = 0; group < lib.numGroups; group++ ) {
            lib.waveforms[group] = stim[group];
            if ( maxDuration < stim[group].duration )
                maxDuration = stim[group].duration;
            if ( minDuration > stim[group].duration )
                minDuration = stim[group].duration;
        }
        lib.pushWaveforms();

        while ( lib.t < maxDuration ) {
            lib.final = lib.t + lib.project.dt() >= maxDuration;
            lib.step();
        }
    }
}
