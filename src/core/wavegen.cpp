#include "wavegen.h"
#include <chrono>
#include <algorithm>
#include <cassert>
#include "cuda_helper.h"

using namespace GeNN_Bridge;

Wavegen::Wavegen(MetaModel &m, const StimulationData &p, const WavegenData &r) :
    p(p),
    r(r),
    m(m),
    blockSize(m.numGroupsPerBlock * (m.adjustableParams.size() + 1)),
    nModels(m.numGroups * (m.adjustableParams.size() + 1)),
    gen(std::chrono::system_clock::now().time_since_epoch().count()),
    stepDist(p.minSteps,p.maxSteps),
    coinflip(0, 1),
    timeDist(p.minStepLength, p.duration - p.minStepLength),
    VDist(p.minVoltage, p.maxVoltage),
    sigmaAdjust(m.adjustableParams.size(), 1.0),
    sigmax(getSigmaMaxima(m))
{
    *simCycles = r.simCycles;
    *clampGain = r.clampGain;
    *accessResistance = r.accessResistance;
}

std::vector<double> Wavegen::getSigmaMaxima(const MetaModel &m)
{
    // Sigmas probably shouldn't exceed 10% of a parameter's range, so let's use that as a maximum:
    constexpr double factor = 0.1;
    std::vector<double> sigmax(m.adjustableParams.size());
    int k = 0;
    for ( const AdjustableParam &p : m.adjustableParams ) {
        if ( p.multiplicative ) {
            // Multiplicative with a range crossing or ending at 0 is daft, but this isn't the place to fail:
            if ( p.max == 0 || p.min == 0 || p.min * p.max < 0 )
                sigmax[k] = factor;
            else
                sigmax[k] = (p.max / p.min) * factor;
        } else {
            sigmax[k] = (p.max - p.min) * factor;
        }
        ++k;
    }
    return sigmax;
}

void Wavegen::permute()
{
    if ( !m.cfg.permute )
        return;

    int stride = 1;

    // If the number of permuted models doesn't fit into thread blocks (very likely),
    // generate a few uncorrelated random parameter sets as padding
    int numPermutedGroups = 1;
    for ( AdjustableParam &p : m.adjustableParams ) {
        numPermutedGroups *= p.wgPermutations + 1;
    }
    int numRandomGroups = m.numGroups - numPermutedGroups;

    for ( AdjustableParam &p : m.adjustableParams ) {
        // First, generate the values:
        std::vector<scalar> values(1, p.initial);
        values.reserve(p.wgPermutations + 1 + numRandomGroups);
        if ( p.wgNormal ) {
            // Draw both permuted and uncorrelated random groups from normal distribution
            std::normal_distribution<scalar> dist(p.initial, p.wgSD);
            for ( int i = 0; i < p.wgPermutations + numRandomGroups; i++ ) {
                scalar v = dist(gen);
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
            std::uniform_real_distribution<scalar> dist(p.min, p.max);
            for ( int i = 0; i < numRandomGroups; i++ ) {
                values.push_back(dist(gen));
            }
        }

        // Populate this parameter, interleaving them in numGroupsPerBlock-periodic fashion
        // This algorithm is designed to maintain sanity rather than memory locality, so it hits each
        // model group in turn, skipping from warp to warp to fill out that group before moving to the next one.
        for ( int group = 0, permutation = p.wgPermutations; group < numPermutedGroups; group++ ) {
            int offset = baseModelIndex(group);
            if ( group % stride == 0)
                permutation = (permutation + 1) % (p.wgPermutations + 1);
            for ( int i = 0, end = m.adjustableParams.size() + 1; i < end; i++ ) {
                p[i*m.numGroupsPerBlock + offset] = values.at(permutation);
            }
        }
        for ( int randomGroup = 0; randomGroup < numRandomGroups; randomGroup++ ) {
            int offset = baseModelIndex(randomGroup+numPermutedGroups);
            for ( int i = 0, end = m.adjustableParams.size() + 1; i < end; i++ ) {
                p[i*m.numGroupsPerBlock + offset] = values.at(p.wgPermutations + 1 + randomGroup);
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
    for ( AdjustableParam &p : m.adjustableParams ) {
        scalar sigma = p.sigma * sigmaAdjust[k] + (p.multiplicative ? 1 : 0);
        for ( int group = 0, paramOffset = ++k * m.numGroupsPerBlock; group < m.numGroups; group++ ) {
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
    for ( int i = 0; i < nModels; i++ )
        getErr[i] = false;
    for ( int i = 0; i < nModels; i++ )
        Vmem[i] = p.baseV;
    *t = 0;
    *iT = 0;
    push();
    while ( *t < r.settleTime ) {
        step();
    }
    pull();
    if ( m.cfg.permute ) {
        // Collect the state variables of every base model, i.e. the tuned version
        settled = std::list<std::vector<scalar>>(m.stateVariables.size(), std::vector<scalar>(m.numGroups));
        auto iter = settled.begin();
        for ( StateVariable &v : m.stateVariables ) {
            for ( int group = 0; group < m.numGroups; group++ ) {
                (*iter)[group] = v[baseModelIndex(group)];
            }
            ++iter;
        }
    } else {
        // Collect the state of one base model
        settled = std::list<std::vector<scalar>>(m.stateVariables.size(), std::vector<scalar>(1));
        auto iter = settled.begin();
        for ( StateVariable &v : m.stateVariables ) {
            (*iter++)[0] = v[0];
        }
    }
}

bool Wavegen::restoreSettled()
{
    if ( settled.empty() )
        return false;

    // Restore to previously found settled state
    auto iter = settled.begin();
    if ( m.cfg.permute ) {
        for ( StateVariable &v : m.stateVariables ) {
            for ( int i = 0; i < nModels; i++ ) {
                int group = i % m.numGroupsPerBlock            // Group index within the block
                        + (i/blockSize) * m.numGroupsPerBlock; // Offset (in group space) of the block this model belongs to
                v[i] = (*iter)[group];
            }
            ++iter;
        }
    } else {
        for ( StateVariable &v : m.stateVariables ) {
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
        getErr[i] = true;
    for ( int i = 0; i < nModels; i++ )
        err[i] = 0;
    *targetParam = -1;
    *t = 0;
    *iT = 0;

    // Generate a set of random waveforms,
    // simulate each (in turn across all model permutations, or in parallel), and collect the
    // per-parameter average deviation from the base model produced by that parameter's detuning.
    std::vector<double> meanParamErr(m.adjustableParams.size());
    int end = m.cfg.permute
            ? r.numSigmaAdjustWaveforms
              // round numSigAdjWaves up to nearest multiple of nModels, and do nGroups waves on each iteration:
            : ((r.numSigmaAdjustWaveforms + nModels - 1) / nModels) * nModels / m.numGroups;
    for ( int i = 0; i < end; i++ ) {
        std::vector<double> sumParamErr(m.adjustableParams.size(), 0);

        // Generate random wave/s
        std::vector<Stimulation> waves;
        if ( m.cfg.permute ) {
            if ( !i )
                waves.resize(1);
            waves[0] = getRandomStim();
        } else {
            if ( !i )
                waves.resize(m.numGroups);
            for ( Stimulation &w : waves )
                w = getRandomStim();
        }

        // Simulate
        restoreSettled();
        stimulate(waves.data());

        // Collect per-parameter error
        PULL(err);
        for ( int j = 0; j < nModels; j++ ) {
            int param = (j % blockSize) / m.numGroupsPerBlock;
            if ( param ) // Collect error for detuned models only
                sumParamErr[param-1] += err[j];
            err[j] = 0;
        }
        // Average on each iteration to maintain precision:
        for ( size_t k = 0; k < m.adjustableParams.size(); k++ ) {
            meanParamErr[k] = (i*meanParamErr[k] + sumParamErr[k]) / (i + 1);
        }
    }

    // Find the median of mean parameter errors:
    double medianErr; {
        std::vector<double> sortedErrs(meanParamErr);
        std::sort(sortedErrs.begin(), sortedErrs.end());
        if ( m.adjustableParams.size() % 2 )
            medianErr = 0.5 * (sortedErrs[m.adjustableParams.size()/2] + sortedErrs[m.adjustableParams.size()/2 - 1]);
        else
            medianErr = sortedErrs[m.adjustableParams.size()/2];
    }

    // Set sigmaAdjust to draw each parameter error average towards the median of parameter error averages
    // Assume a more-or-less linear relationship, where doubling the sigma roughly doubles the deviation.
    // This is a simplification, but it should work well enough for small sigmas and small increments thereof.
    double maxExcess = 1;
    for ( size_t k = 0; k < m.adjustableParams.size(); k++ ) {
        sigmaAdjust[k] *= medianErr / meanParamErr[k];
        if ( sigmaAdjust[k] > sigmax[k] ) {
            double excess = sigmaAdjust[k] / sigmax[k];
            if ( excess > maxExcess )
                maxExcess = excess;
        }
    }
    // Ensure that no sigmaAdjust exceeds its sigmax boundaries:
    if ( maxExcess > 1 )
        for ( double &adj : sigmaAdjust )
            adj /= maxExcess;
}

void Wavegen::stimulate(const Stimulation *stim)
{
    // NYI

}

void Wavegen::search()
{

}

Stimulation Wavegen::getRandomStim()
{
    Stimulation I;
    int failedPos, failedAgain = 0;
    I.baseV = p.baseV;
    I.duration = p.duration;
    int n = stepDist(gen);
tryagain:
    failedPos = 0;
    if ( failedAgain++ > 2*p.maxSteps ) {
        failedAgain = 0;
        --n;
    }
    I.steps.clear();
    for ( int i = 0; i < n; i++ ) {
        double t = timeDist(gen);
        for ( Stimulation::Step s : I.steps ) {
            if ( fabs(s.t - t) < p.minStepLength ) {
                ++failedPos;
                break;
            } else {
                failedPos = 0;
            }
        }
        if ( failedPos ) {
            if ( failedPos > 2*p.maxSteps )
                goto tryagain;
            --i;
            continue;
        }
        I.steps.push_back(Stimulation::Step{t, VDist(gen), (bool)(coinflip(gen))});
    }
    return I;
}
