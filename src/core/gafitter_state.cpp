#include "gafitter.h"
#include "session.h"

void GAFitter::setup()
{
    int nParams = lib.adjustableParams.size();
    int nStims = astims.size();

    if ( output.resume.population.empty() ) {
        DEMethodUsed.assign(lib.NMODELS/2, 0);
        DEMethodSuccess.assign(4, 0);
        DEMethodFailed.assign(4, 0);
        DEpX.assign(lib.NMODELS/2, 0);
    } else {
        DEMethodUsed = output.resume.DEMethodUsed;
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
    if ( obsSource == Wavegen::cluster_action || obsSource == Wavegen::bubble_action ) {
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
