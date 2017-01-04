#include "wavegen.h"
#include <algorithm>
#include <cassert>
#include "cuda_helper.h"

Wavegen::Wavegen(MetaModel &m, const std::string &dir, const StimulationData &p, const WavegenData &r, const RunData &rund) :
    WavegenConstructor(m, dir, r),
    p(p),
    blockSize(numGroupsPerBlock * (adjustableParams.size() + 1)),
    nModels(numGroups * (adjustableParams.size() + 1)),
    RNG(),
    sigmaAdjust(adjustableParams.size(), 1.0),
    sigmax(getSigmaMaxima()),
    mapeStats(r.historySize, mapeArchive.end())
{
    setRunData(rund);
}

void Wavegen::setRunData(const RunData &r)
{
    rund = r;
    simCycles = r.simCycles;
    clampGain = r.clampGain;
    accessResistance = r.accessResistance;
}

std::vector<double> Wavegen::getSigmaMaxima()
{
    // Sigmas probably shouldn't exceed 10% of a parameter's range, so let's use that as a maximum:
    constexpr double factor = 0.1;
    std::vector<double> sigmax(adjustableParams.size());
    int k = 0;
    for ( const AdjustableParam &p : adjustableParams ) {
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

void Wavegen::permute()
{
    if ( !r.permute )
        return;

    int stride = 1;

    // If the number of permuted models doesn't fit into thread blocks (very likely),
    // generate a few uncorrelated random parameter sets as padding
    int numPermutedGroups = 1;
    for ( AdjustableParam &p : adjustableParams ) {
        numPermutedGroups *= p.wgPermutations + 1;
    }
    int numRandomGroups = numGroups - numPermutedGroups;

    for ( AdjustableParam &p : adjustableParams ) {
        // First, generate the values:
        std::vector<scalar> values(1, p.initial);
        values.reserve(p.wgPermutations + 1 + numRandomGroups);
        if ( p.wgNormal ) {
            // Draw both permuted and uncorrelated random groups from normal distribution
            for ( int i = 0; i < p.wgPermutations + numRandomGroups; i++ ) {
                scalar v = RNG.variate<scalar>(p.initial, p.wgSD);
                if ( v > p.max )
                    v = p.max;
                else if ( v < p.min )
                    v = p.min;
                values.push_back(v);
            }
        } else {
            // Permuted groups: Space evenly over the parameter range
            scalar step = (p.max - p.min) / (p.wgPermutations + 1);
            for ( int i = 1; i < p.wgPermutations + 1; i++ ) {
                values.push_back(p.min + i*step);
            }
            // Random groups: Draw from uniform distribution
            for ( int i = 0; i < numRandomGroups; i++ ) {
                values.push_back(RNG.uniform(p.min, p.max));
            }
        }

        // Populate this parameter, interleaving them in numGroupsPerBlock-periodic fashion
        // This algorithm is designed to maintain sanity rather than memory locality, so it hits each
        // model group in turn, skipping from warp to warp to fill out that group before moving to the next one.
        for ( int group = 0, permutation = p.wgPermutations; group < numPermutedGroups; group++ ) {
            int offset = baseModelIndex(group);
            if ( group % stride == 0)
                permutation = (permutation + 1) % (p.wgPermutations + 1);
            for ( int i = 0, end = adjustableParams.size() + 1; i < end; i++ ) {
                p[i*numGroupsPerBlock + offset] = values.at(permutation);
            }
        }
        for ( int randomGroup = 0; randomGroup < numRandomGroups; randomGroup++ ) {
            int offset = baseModelIndex(randomGroup+numPermutedGroups);
            for ( int i = 0, end = adjustableParams.size() + 1; i < end; i++ ) {
                p[i*numGroupsPerBlock + offset] = values.at(p.wgPermutations + 1 + randomGroup);
            }
        }

        // Permutation stride starts out at 1 and increases from one parameter to the next
        stride *= p.wgPermutations + 1;
    }
    settled.clear();
}

void Wavegen::detune()
{
    int k = 0;
    for ( AdjustableParam &p : adjustableParams ) {
        scalar sigma = p.sigma * sigmaAdjust[k] + (p.multiplicative ? 1 : 0);
        for ( int group = 0, paramOffset = ++k * numGroupsPerBlock; group < numGroups; group++ ) {
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
    I.duration = r.settleTime;
    I.baseV = p.baseV;
    I.clear();
    for ( int group = 0; group < numGroups; group++ )
        waveforms[group] = I;
    pushWaveforms();
    getErr = false;
    t = 0;
    iT = 0;
    push();
    while ( t < r.settleTime ) {
        step();
    }
    pull();
    if ( r.permute ) {
        // Collect the state variables of every base model, i.e. the tuned version
        settled = std::list<std::vector<scalar>>(stateVariables.size(), std::vector<scalar>(numGroups));
        auto iter = settled.begin();
        for ( StateVariable &v : stateVariables ) {
            for ( int group = 0; group < numGroups; group++ ) {
                (*iter)[group] = v[baseModelIndex(group)];
            }
            ++iter;
        }

        // Provide some auditing info
        std::vector<scalar> V;
        int valid = 0;
        for ( int group = 0; group < numGroups; group++ ) {
            scalar const& v = stateVariables[0].v[baseModelIndex(group)];
            V.push_back(v);
            if ( v > p.baseV-5 && v < p.baseV+5 )
                valid++;
        }
        std::sort(V.begin(), V.end());
        std::cout << "Settled all permuted models to holding potential of " << p.baseV << " mV for "
                  << r.settleTime << " ms." << std::endl;
        std::cout << "Median achieved potential: " << V[numGroups/2] << " mV (95% within [" << V[numGroups/20]
                  << " mV, " << V[numGroups/20*19] << " mV]), " << valid << "/" << numGroups
                  << " models within +-5 mV of holding." << std::endl;
    } else {
        // Collect the state of one base model
        settled = std::list<std::vector<scalar>>(stateVariables.size(), std::vector<scalar>(1));
        auto iter = settled.begin();
        for ( StateVariable &v : stateVariables ) {
            (*iter++)[0] = v[0];
        }

        std::cout << "Settled base model to holding potential of " << p.baseV << " mV for " << r.settleTime << " ms, "
                  << "achieved " << (*settled.begin())[0] << " mV." << std::endl;
    }
}

bool Wavegen::restoreSettled()
{
    if ( settled.empty() )
        return false;

    // Restore to previously found settled state
    auto iter = settled.begin();
    if ( r.permute ) {
        for ( StateVariable &v : stateVariables ) {
            for ( int i = 0; i < nModels; i++ ) {
                int group = i % numGroupsPerBlock            // Group index within the block
                        + (i/blockSize) * numGroupsPerBlock; // Offset (in group space) of the block this model belongs to
                v[i] = (*iter)[group];
            }
            ++iter;
        }
    } else {
        for ( StateVariable &v : stateVariables ) {
            for ( int i = 0; i < nModels; i++ ) {
                v[i] = (*iter)[0];
            }
            ++iter;
        }
    }
    push();
    return true;
}

void Wavegen::adjustSigmas()
{
    if ( settled.empty() )
        settle();
    detune();
    for ( int i = 0; i < nModels; i++ )
        err[i] = 0;
    getErr = true;
    targetParam = -1;

    // Generate a set of random waveforms,
    // simulate each (in turn across all model permutations, or in parallel), and collect the
    // per-parameter average deviation from the base model produced by that parameter's detuning.
    std::vector<double> sumParamErr(adjustableParams.size(), 0);
    std::vector<Stimulation> waves;
    int end = r.permute
            ? r.numSigmaAdjustWaveforms
              // round numSigAdjWaves up to nearest multiple of nGroups to fully occupy each iteration:
            : ((r.numSigmaAdjustWaveforms + numGroups - 1) / numGroups);
    for ( int i = 0; i < end; i++ ) {

        // Generate random wave/s
        if ( r.permute ) {
            if ( !i )
                waves.resize(1);
            waves[0] = getRandomStim();
        } else {
            if ( !i )
                waves.resize(numGroups);
            for ( Stimulation &w : waves )
                w = getRandomStim();
        }

        // Simulate
        restoreSettled();
        stimulate(waves);

        // Collect per-parameter error
        pullErr();
        for ( int j = 0; j < nModels; j++ ) {
            int param = (j % blockSize) / numGroupsPerBlock;
            if ( param && !isnan(err[j]) ) // Collect error for stable detuned models only
                sumParamErr[param-1] += err[j];
            err[j] = 0;
        }
    }

    std::vector<double> meanParamErr(sumParamErr);
    for ( double & e : meanParamErr ) {
        e /= end * numGroups * p.duration/m.cfg.dt * rund.simCycles;
    }

    // Find the median of mean parameter errors:
    double medianErr; {
        std::vector<double> sortedErrs(meanParamErr);
        std::sort(sortedErrs.begin(), sortedErrs.end());
        if ( adjustableParams.size() % 2 )
            medianErr = 0.5 * (sortedErrs[adjustableParams.size()/2] + sortedErrs[adjustableParams.size()/2 - 1]);
        else
            medianErr = sortedErrs[adjustableParams.size()/2];
    }

    // Set sigmaAdjust to draw each parameter error average towards the median of parameter error averages
    // Assume a more-or-less linear relationship, where doubling the sigma roughly doubles the deviation.
    // This is a simplification, but it should work well enough for small sigmas and small increments thereof.
    double maxExcess = 1;
    for ( size_t k = 0; k < adjustableParams.size(); k++ ) {
        sigmaAdjust[k] *= medianErr / meanParamErr[k];
        if ( sigmaAdjust[k] * adjustableParams[k].sigma > sigmax[k] ) {
            double excess = sigmaAdjust[k] * adjustableParams[k].sigma / sigmax[k];
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
    for ( int i = 0; i < (int)adjustableParams.size(); i++ )
        std::cout << adjustableParams[i].name << ":\t" << meanParamErr[i] << '\t'
                  << sigmaAdjust[i] << '\t' << adjustableParams[i].sigma*sigmaAdjust[i] << std::endl;
}

void Wavegen::stimulate(const std::vector<Stimulation> &stim)
{
    t = 0;
    iT = 0;
    if ( r.permute ) {
        const Stimulation &s = stim.at(0);
        for ( int group = 0; group < numGroups; group++ )
            waveforms[group] = s;
        pushWaveforms();
        while ( t < s.duration ) {
            final = t + m.cfg.dt >= s.duration;
            step();
        }
    } else { //-------------- !m.cfg.permute ------------------------------------------------------------
        assert((int)stim.size() >= numGroups);
        double maxDuration = 0.0, minDuration = stim[0].duration;
        for ( int group = 0; group < numGroups; group++ ) {
            waveforms[group] = stim[group];
            if ( maxDuration < stim[group].duration )
                maxDuration = stim[group].duration;
            if ( minDuration > stim[group].duration )
                minDuration = stim[group].duration;
        }
        pushWaveforms();

        while ( t < maxDuration ) {
            final = t + m.cfg.dt >= maxDuration;
            step();
        }
    }
}
