#include "wavegen.h"
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
    RNG(),
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
        if ( p.sigma > sigmax[k] )
            sigmax[k] = p.sigma;
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
    for ( int i = 0; i < nModels; i++ )
        Vramp[i] = 0.0;
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

        // Provide some auditing info
        std::vector<scalar> V;
        int valid = 0;
        for ( int group = 0; group < m.numGroups; group++ ) {
            scalar const& v = m.stateVariables[0].v[baseModelIndex(group)];
            V.push_back(v);
            if ( v > -65 && v < -55 )
                valid++;
        }
        std::sort(V.begin(), V.end());
        std::cout << "Settled all permuted models to holding potential of " << p.baseV << " mV for "
                  << r.settleTime << " ms." << std::endl;
        std::cout << "Median achieved potential: " << V[m.numGroups/2] << " mV (95% within [" << V[m.numGroups/20]
                  << " mV, " << V[m.numGroups/20*19] << " mV]), " << valid << "/" << m.numGroups
                  << " models within +-5 mV of holding." << std::endl;
    } else {
        // Collect the state of one base model
        settled = std::list<std::vector<scalar>>(m.stateVariables.size(), std::vector<scalar>(1));
        auto iter = settled.begin();
        for ( StateVariable &v : m.stateVariables ) {
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
    std::vector<double> sumParamErr(m.adjustableParams.size(), 0);
    std::vector<Stimulation> waves;
    int end = m.cfg.permute
            ? r.numSigmaAdjustWaveforms
              // round numSigAdjWaves up to nearest multiple of nGroups to fully occupy each iteration:
            : ((r.numSigmaAdjustWaveforms + m.numGroups - 1) / m.numGroups);
    for ( int i = 0; i < end; i++ ) {

        // Generate random wave/s
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
        stimulate(waves);

        // Collect per-parameter error
        PULL(err);
        for ( int j = 0; j < nModels; j++ ) {
            int param = (j % blockSize) / m.numGroupsPerBlock;
            if ( param && !isnan(err[j]) ) // Collect error for stable detuned models only
                sumParamErr[param-1] += err[j];
            err[j] = 0;
        }
    }

    std::vector<double> meanParamErr(sumParamErr);
    for ( double & e : meanParamErr ) {
        e /= end * m.numGroups * p.duration/m.cfg.dt * r.simCycles;
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
        if ( sigmaAdjust[k] * m.adjustableParams[k].sigma > sigmax[k] ) {
            double excess = sigmaAdjust[k] * m.adjustableParams[k].sigma / sigmax[k];
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
    for ( int i = 0; i < (int)m.adjustableParams.size(); i++ )
        std::cout << m.adjustableParams[i].name << ":\t" << meanParamErr[i] << '\t'
                  << sigmaAdjust[i] << '\t' << m.adjustableParams[i].sigma*sigmaAdjust[i] << std::endl;
}

void Wavegen::stimulate(const std::vector<Stimulation> &stim)
{
    scalar rampDelta;
    std::vector<Stimulation::Step> finalSteps;
    *t = 0;
    *iT = 0;
    if ( m.cfg.permute ) {
        const Stimulation &s = stim.at(0);
        auto iter = s.steps.begin();
        short obs = *targetParam < 0 ? 2 : 0;
        finalSteps.push_back(Stimulation::Step {s.duration, s.baseV, false} );
        rampDelta = iter->ramp ? (iter->V - s.baseV) / (r.simCycles * iter->t / m.cfg.dt) : 0.0;
        for ( int i = 0; i < nModels; i++ )
            Vmem[i] = s.baseV;
        for ( int i = 0; i < nModels; i++ )
            Vramp[i] = rampDelta;
        if ( *targetParam >= 0 ) {
            for ( int i = 0; i < nModels; i++ )
                getErr[i] = false;
            PUSH(getErr);
        }
        PUSH(Vmem);
        PUSH(Vramp);
        while ( *t < s.duration ) {
            if ( iter->t <= *t ) {
                for ( int i = 0; i < nModels; i++ )
                    Vmem[i] = iter->V;
                bool wasRamp = iter->ramp;
                if ( ++iter == s.steps.end() )
                    iter = finalSteps.begin();
                if ( wasRamp || iter->ramp ) {
                    rampDelta = iter->ramp
                            ? (iter->V - (iter-1)->V) / (r.simCycles * (iter->t - (iter-1)->t) / m.cfg.dt)
                            : 0.0;
                    for ( int i = 0; i < nModels; i++ )
                        Vramp[i] = rampDelta;
                    PUSH(Vramp);
                }
                PUSH(Vmem);
            }
            if ( !obs && *t >= s.tObsBegin ) {
                ++obs;
                for ( int i = 0; i < nModels; i++ )
                    getErr[i] = true;
                PUSH(getErr);
            } else if ( obs == 1 && *t >= s.tObsEnd ) {
                ++obs;
                for ( int i = 0; i < nModels; i++ )
                    getErr[i] = false;
                PUSH(getErr);
            }
            step();
        }
    } else { //-------------- !m.cfg.permute ------------------------------------------------------------
        assert((int)stim.size() >= m.numGroups);
        std::vector<std::vector<Stimulation::Step>::const_iterator> iter;
        std::vector<short> obs;
        iter.resize(m.numGroups);
        obs.resize(m.numGroups, *targetParam < 0 ? 2 : 0);
        double maxDuration = 0.0, minDuration = stim[0].duration;
        for ( int group = 0; group < m.numGroups; group++ ) {
            iter[group] = stim[group].steps.begin();
            if ( maxDuration < stim[group].duration )
                maxDuration = stim[group].duration;
            if ( minDuration > stim[group].duration )
                minDuration = stim[group].duration;
            rampDelta = iter[group]->ramp
                    ? (iter[group]->V - stim[group].baseV) / (r.simCycles * iter[group]->t / m.cfg.dt)
                    : 0.0;
            int offset = baseModelIndex(group);
            for ( int i = 0, end = m.adjustableParams.size() + 1; i < end; i++ ) {
                Vmem[i*m.numGroupsPerBlock + offset] = iter[group]->V;
                Vramp[i*m.numGroupsPerBlock + offset] = rampDelta;
            }
        }
        if ( *targetParam >= 0 ) {
            for ( int i = 0; i < nModels; i++ )
                getErr[i] = false;
            PUSH(getErr);
        }
        PUSH(Vmem);
        PUSH(Vramp);
        if ( minDuration == maxDuration ) {
            // Since this step is never actually reached, don't worry about individual voltages:
            finalSteps.push_back(Stimulation::Step {maxDuration, stim[0].baseV, false} );
        } else {
            finalSteps.resize(2*m.numGroups);
            for ( int group = 0; group < m.numGroups; group++ ) {
                finalSteps[2*group] = Stimulation::Step {stim[group].duration, stim[group].baseV, false};
                finalSteps[2*group+1] = Stimulation::Step {maxDuration, stim[group].baseV, false};
            }
        }
        while ( *t < maxDuration ) {
            bool pushVmem = false,
                 pushRamp = false,
                 pushGetErr = false;
            for ( int group = 0; group < m.numGroups; group++ ) {
                if ( iter[group]->t <= *t ) {
                    int offset = baseModelIndex(group);
                    for ( int i = 0, end = m.adjustableParams.size() + 1; i < end; i++ ) {
                        Vmem[i*m.numGroupsPerBlock + offset] = iter[group]->V;
                    }
                    bool wasRamp = iter[group]->ramp;
                    if ( ++iter[group] == stim[group].steps.end() ) {
                        if ( minDuration == maxDuration )
                            iter[group] = finalSteps.begin();
                        else
                            iter[group] = finalSteps.begin() + 2*group;
                    }
                    if ( wasRamp || iter[group]->ramp ) {
                        rampDelta = iter[group]->ramp
                                ? (iter[group]->V - (iter[group]-1)->V) / (r.simCycles * (iter[group]->t - (iter[group]-1)->t) / m.cfg.dt)
                                : 0.0;
                        for ( int i = 0, end = m.adjustableParams.size() + 1; i < end; i++ ) {
                            Vramp[i*m.numGroupsPerBlock + offset] = rampDelta;
                        }
                        pushRamp = true;
                    }
                    pushVmem = true;
                }
                if ( !obs[group] && *t >= stim[group].tObsBegin ) {
                    ++obs[group];
                    for ( int i = 0, end = m.adjustableParams.size() + 1, offset = baseModelIndex(group); i < end; i++ )
                        getErr[i*m.numGroupsPerBlock + offset] = true;
                    pushGetErr = true;
                } else if ( obs[group] == 1 && *t >= stim[group].tObsEnd ) {
                    ++obs[group];
                    for ( int i = 0, end = m.adjustableParams.size() + 1, offset = baseModelIndex(group); i < end; i++ )
                        getErr[i*m.numGroupsPerBlock + offset] = false;
                    pushGetErr = true;
                }
            }
            // Apparently cudaMemcpy has a high overhead, so group- or blockwise transfers might not be any faster than this:
            if ( pushVmem )
                PUSH(Vmem);
            if ( pushRamp )
                PUSH(Vramp);
            if ( pushGetErr )
                PUSH(getErr);
            step();
        }
    }
}

void Wavegen::search()
{

}
