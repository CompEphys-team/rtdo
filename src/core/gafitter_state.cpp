/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "gafitter.h"
#include "session.h"
#include "clustering.h"

void GAFitter::setup(bool ad_hoc_stims)
{
    int nParams = lib.adjustableParams.size();
    int nStims = astims.size();

    DEMethodUsed.assign(lib.NMODELS/2, 0);
    DEpXUsed.assign(lib.NMODELS/2, 0);
    if ( output.resume.population.empty() ) {
        DEMethodSuccess.assign(4, 1);
        DEMethodFailed.assign(4, 0);
        DEpX.assign(3, 0.5);
    } else {
        DEMethodSuccess = output.resume.DEMethodSuccess;
        DEMethodFailed = output.resume.DEMethodFailed;
        DEpX = output.resume.DEpX;
    }

    bias.assign(nStims, 0);

    epoch = targetStim = 0;
    targetStim = findNextStim();

    stims.resize(nStims);
    obs.resize(nStims);
    baseF.resize(nStims, std::vector<double>(nParams, 0));

    QString obsSource = QString::fromStdString(settings.obsSource);
    if ( ad_hoc_stims ) {
        iObservations obsAll = {{}, {}};
        obsAll.stop[0] = iStimData(session.stimulationData(), session.runData().dt).iDuration;
        obs.assign(nStims, obsAll);
    } else if ( obsSource == Wavegen::cluster_action || obsSource == Wavegen::bubble_action ) {
        auto elites = session.wavegen().findObservations(output.stimSource.iStimulations(session.runData().dt), obsSource);
        std::vector<Stimulation> astims_ordered(nStims);
        for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
            scalar bestFitness = 0;
            size_t bestStimIdx = 0;
            for ( size_t obsIdx = 0; obsIdx < elites[stimIdx].size(); obsIdx++ ) {
                const MAPElite &el = elites[stimIdx][obsIdx];
                if ( el.fitness > bestFitness ) {
                    bestFitness = el.fitness;
                    bestStimIdx = obsIdx;
                }
            }
            stims[stimIdx] = *elites[stimIdx][bestStimIdx].wave;
            obs[stimIdx] = elites[stimIdx][bestStimIdx].obs;
            for ( int i = 0; i < nParams; i++ )
                baseF[stimIdx][i] = elites[stimIdx][bestStimIdx].deviations[i];
            astims_ordered[stimIdx] = astims[bestStimIdx];
        }
        using std::swap;
        swap(astims, astims_ordered);
    } else if ( obsSource == "random" ) {
        std::vector<int> idx(nStims);
        for ( int i = 0; i < nStims; i++ )
            idx[i] = i;
        session.RNG.shuffle(idx);

        std::vector<MAPElite> elites = output.stimSource.elites();
        std::vector<Stimulation> astims_orig(astims);
        for ( int i = 0; i < nStims; i++ ) {
            astims[i] = astims_orig[idx[i]];
            stims[i] = *elites[idx[i]].wave;

            // Create random observation with similar statistics to this target's obs (no index shuffling here!)
            iObservations ref = elites[i].obs;
            std::vector<int> chunks;
            for ( size_t i = 0; i < iObservations::maxObs && ref.stop[i] > 0; i++ )
                chunks.push_back(ref.stop[i] - ref.start[i]);
            std::sort(chunks.rbegin(), chunks.rend()); // largest first

            std::vector<std::pair<int, int>> observables = observeNoSteps(stims[i], session.wavegenData().cluster.blank/session.runData().dt);
            std::vector<std::pair<int, int>> observations;
            for ( int chunk : chunks ) {
                std::vector<int> possibleLocations;
                ++chunk;
                do {
                    --chunk;
                    for ( size_t j = 0; j < observables.size(); j++ )
                        if ( observables[j].second - observables[j].first >= chunk )
                            possibleLocations.push_back(j);
                } while ( possibleLocations.empty() );
                int loc = session.RNG.pick(possibleLocations);
                int leeway = observables[loc].second - observables[loc].first - chunk;
                int start = observables[loc].first + session.RNG.uniform<int>(0, leeway);
                observations.push_back(std::make_pair(start, start+chunk));

                if ( leeway == 0 ) {
                    observables.erase(observables.begin() + loc);
                } else if ( start == 0 ) {
                    observables[loc].first += chunk;
                } else if ( start == leeway ) {
                    observables[loc].second -= chunk;
                } else {
                    observables.insert(observables.begin() + loc + 1, std::make_pair(start+chunk, observables[loc].second));
                    observables[loc].second = start;
                }
            }
            std::sort(observations.begin(), observations.end());
            obs[i] = {{},{}};
            for ( size_t j = 0; j < observations.size(); j++ ) {
                obs[i].start[j] = observations[j].first;
                obs[i].stop[j] = observations[j].second;
            }
        }

        if ( settings.mutationSelectivity == 1 ) {
            std::vector<MAPElite> posthoc = session.wavegen().evaluatePremade(stims, obs);
            for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ )
                for ( int i = 0; i < nParams; i++ )
                    baseF[stimIdx][i] = posthoc[stimIdx].deviations[i];
        }
    } else {
        std::vector<int> needPosthocEval;
        int stimIdx = 0;
        for ( const MAPElite &el : output.stimSource.elites() ) {
            stims[stimIdx] = *el.wave;
            obs[stimIdx] = el.obs;

            double sumDev = 0;
            for ( const scalar &dev : el.deviations )
                sumDev += dev;
            if ( sumDev == 0 ) {
                needPosthocEval.push_back(stimIdx);
            } else {
                for ( int i = 0; i < nParams; i++ )
                    baseF[stimIdx][i] = el.deviations[i];
            }
            ++stimIdx;
        }

        if ( settings.mutationSelectivity == 1 && !needPosthocEval.empty() ) {
            std::vector<MAPElite> posthoc = session.wavegen().evaluatePremade(stims, obs);
            for ( int stimIdx : needPosthocEval )
                for ( int i = 0; i < nParams; i++ )
                    baseF[stimIdx][i] = posthoc[stimIdx].deviations[i];
        }
    }

    if ( settings.mutationSelectivity == 2 ) { // note: nStims==nParams
        baseF.assign(nStims, std::vector<double>(nParams, 0));
        for ( int i = 0; i < nParams; i++ )
            baseF[i][i] = 1;
    } else if ( settings.mutationSelectivity == 0 ) {
        baseF.assign(nStims, std::vector<double>(nParams, 1));
    }

    errNorm.resize(nStims);
    for ( int stimIdx = 0; stimIdx < nStims; stimIdx++ )
        errNorm[stimIdx] = obs[stimIdx].duration();

    output.stims = QVector<iStimulation>::fromStdVector(stims);
    output.obs = QVector<iObservations>::fromStdVector(obs);
    output.baseF.resize(nStims);
    for ( int i = 0; i < nStims; i++ )
        output.baseF[i] = QVector<double>::fromStdVector(baseF[i]);
}

void GAFitter::populate()
{
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        if ( settings.constraints[i] >= 2 ) { // 2,Fixed, 3,target, 4,resume:final, 5,resume:mean
            double value = settings.constraints[i] == 2 ? settings.fixedValue[i] : output.targets[i];
            if ( settings.constraints[i] == 4 ) // resume:final
                if ( output.final ) // If available
                    value = output.finalParams[i];
                else if ( output.epochs > 0 ) // First fallback: preceding population's best
                    value = output.params[output.epochs-1][i];
                // Second fallback: target
            else if ( settings.constraints[i] == 5 && !output.resume.population.empty() ) { // resume:mean, if available. Fallback to target
                value = 0;
                for ( size_t j = 0; j < lib.NMODELS; j++ )
                    value += output.resume.population[i][j];
                value /= lib.NMODELS;
            }
            for ( size_t j = 0; j < lib.NMODELS; j++ )
                lib.adjustableParams[i][j] = value;
        } else {
            scalar min, max;
            if ( settings.constraints[i] == 0 ) {
                min = lib.adjustableParams[i].min;
                max = lib.adjustableParams[i].max;
            } else {
                min = settings.min[i];
                max = settings.max[i];
            }
            if ( output.resume.population.empty() ) {
                scalar sig = (max-min)/6;
                scalar mid = min + (max-min)/2;
                scalar x;
                for ( size_t j = 0; j < lib.NMODELS; j++ ) {
                    do {
                        x = session.RNG.variate<scalar>(mid, sig);
                    } while ( x < min || x > max );
                    lib.adjustableParams[i][j] = x;
                }
            } else {
                for ( size_t j = 0; j < lib.NMODELS; j++ )
                    lib.adjustableParams[i][j] =
                            std::min(max, std::max(min, output.resume.population[i][j]));
            }
        }
    }
    lib.push();
}
