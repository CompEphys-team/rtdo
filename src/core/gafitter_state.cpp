#include "gafitter.h"
#include "session.h"

void GAFitter::setup()
{
    int nParams = lib.adjustableParams.size();

    if ( output.resume.population.empty() ) {
        DEMethodUsed.assign(lib.NMODELS/2, 0);
        DEMethodSuccess.assign(4, 0);
        DEMethodFailed.assign(4, 0);
        DEpX.assign(lib.NMODELS/2, 0);

        bias.assign(nParams, 0);
    } else {
        DEMethodUsed = output.resume.DEMethodUsed;
        DEMethodSuccess = output.resume.DEMethodSuccess;
        DEMethodFailed = output.resume.DEMethodFailed;
        DEpX = output.resume.DEpX;

        bias = output.resume.bias;
    }

    epoch = targetParam = 0;
    targetParam = findNextStim();

    stims.resize(nParams);
    obs.resize(nParams);
    baseF.resize(nParams, std::vector<double>(nParams, 0));

    QString obsSource = QString::fromStdString(settings.obsSource);
    if ( obsSource == Wavegen::cluster_action || obsSource == Wavegen::bubble_action ) {
        auto elites = session.wavegen().findObservations(output.stimSource.iStimulations(session.runData().dt), obsSource);
        std::vector<Stimulation> astims_ordered(nParams);
        for ( int paramIdx = 0; paramIdx < nParams; paramIdx++ ) {
            scalar bestFitness = 0;
            size_t bestStimIdx = 0;
            for ( size_t stimIdx = 0; stimIdx < elites[paramIdx].size(); stimIdx++ ) {
                const MAPElite &el = elites[paramIdx][stimIdx];
                if ( el.fitness > bestFitness ) {
                    bestFitness = el.fitness;
                    bestStimIdx = stimIdx;
                }
            }
            stims[paramIdx] = *elites[paramIdx][bestStimIdx].wave;
            obs[paramIdx] = elites[paramIdx][bestStimIdx].obs;
            for ( int i = 0; i < nParams; i++ )
                baseF[paramIdx][i] = elites[paramIdx][bestStimIdx].deviations[i];
            astims_ordered[paramIdx] = astims[bestStimIdx];
        }
        using std::swap;
        swap(astims, astims_ordered);
    } else {
        std::vector<int> needPosthocEval;
        int paramIdx = 0;
        for ( const MAPElite &el : output.stimSource.elites() ) {
            stims[paramIdx] = *el.wave;
            obs[paramIdx] = el.obs;

            double sumDev = 0;
            for ( const scalar &dev : el.deviations )
                sumDev += dev;
            if ( sumDev == 0 ) {
                needPosthocEval.push_back(paramIdx);
            } else {
                for ( int i = 0; i < nParams; i++ )
                    baseF[paramIdx][i] = el.deviations[i];
            }
            ++paramIdx;
        }

        if ( settings.mutationSelectivity == 1 && !needPosthocEval.empty() ) {
            std::vector<MAPElite> posthoc = session.wavegen().evaluatePremade(stims, obs);
            for ( int paramIdx : needPosthocEval )
                for ( int i = 0; i < nParams; i++ )
                    baseF[paramIdx][i] = posthoc[paramIdx].deviations[i];
        }
    }

    if ( settings.mutationSelectivity == 2 ) {
        baseF.assign(nParams, std::vector<double>(nParams, 0));
        for ( int i = 0; i < nParams; i++ )
            baseF[i][i] = 1;
    } else if ( settings.mutationSelectivity == 0 ) {
        baseF.assign(nParams, std::vector<double>(nParams, 1));
    }

    errNorm.resize(nParams);
    for ( int paramIdx = 0; paramIdx < nParams; paramIdx++ )
        errNorm[paramIdx] = obs[paramIdx].duration();

    output.stims = QVector<iStimulation>::fromStdVector(stims);
    output.obs = QVector<iObservations>::fromStdVector(obs);
    output.baseF.resize(nParams);
    for ( int i = 0; i < nParams; i++ )
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
