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
    if ( output.resume.population.empty() ) {
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
            if ( settings.constraints[i] == 2 || settings.constraints[i] == 3 ) { // Fixed value
                scalar value = settings.constraints[i] == 2 ? settings.fixedValue[i] : output.targets[i];
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
                for ( size_t j = 0; j < lib.NMODELS; j++ ) {
                    lib.adjustableParams[i][j] = session.RNG.uniform<scalar>(min, max);
                }
            }
        }
    } else {
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ )
            for ( size_t j = 0; j < lib.NMODELS; j++ )
                lib.adjustableParams[i][j] = output.resume.population[i][j];
    }
    lib.push();
}
