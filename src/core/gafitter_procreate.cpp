#include "gafitter.h"
#include "session.h"

void GAFitter::procreate()
{
    std::vector<errTupel> p_err(lib.NMODELS);
    for ( size_t i = 0; i < p_err.size(); i++ ) {
        p_err[i].idx = i;
        p_err[i].err = lib.summary[i];
    }
    std::sort(p_err.begin(), p_err.end(), &errTupelSort);

    output.error[epoch] = std::sqrt(p_err[0].err / errNorm[targetParam]); // RMSE
    output.targetParam[epoch] = targetParam;
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.params[epoch][i] = lib.adjustableParams[i][p_err[0].idx];
    }

    targetParam = findNextStim();

    double F = settings.decaySigma ? settings.sigmaInitial * std::exp2(-double(epoch)/settings.sigmaHalflife) : 1;

    // Mutate
    for ( size_t i = p_err.size()-settings.nReinit-1; i >= settings.nElite; i-- ) {
        // Bias reproductions towards the elite in roughly linear fashion by restricting the choice range
        size_t targetSource = session.RNG.uniform<size_t>(0, i-settings.nElite);
        size_t source = targetSource;

        size_t otherSource = targetSource;
        unsigned int sourceSelect = 0;
        if ( settings.crossover > 0 && session.RNG.uniform<double>(0,1) < settings.crossover )
            otherSource = session.RNG.uniform<size_t>(0, i-settings.nElite);

        for ( size_t iParam = 0; iParam < lib.adjustableParams.size(); iParam++ ) {

            // Ignore fixed-value parameters
            if ( settings.constraints[iParam] >= 2 )
                continue;

            // Parameter-wise crossover, implemented with minimal RNG use
            if ( targetSource != otherSource ) {
                if ( iParam % (8*sizeof sourceSelect) == 0 )
                    sourceSelect = session.RNG.uniform<unsigned int>(0, ~0);
                source = (sourceSelect & 0x1) ? otherSource : targetSource;
                sourceSelect = sourceSelect >> 1;
            }

            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( settings.mutationSelectivity < 2 || iParam == targetParam ) {
                // Mutate target param
                scalar sigma = settings.constraints[iParam] == 1 ? settings.sigma[iParam] : lib.adjustableParams[iParam].sigma;
                if ( !p.multiplicative )
                    p[p_err[i].idx] = session.RNG.variate<scalar, std::normal_distribution>(
                                        p[p_err[source].idx], baseF[targetParam][iParam] * F * sigma);
                else if ( int(iParam) < lib.model.nNormalAdjustableParams ) {
                    scalar factor = -1;
                    while ( factor < 0 )
                        factor = session.RNG.variate<scalar, std::normal_distribution>(
                                  1, baseF[targetParam][iParam] * F * sigma);
                    p[p_err[i].idx] = p[p_err[source].idx] * factor;
                } else {
                    p[p_err[i].idx] = p[p_err[source].idx] *
                            ( session.RNG.variate<scalar, std::uniform_real_distribution>(0, settings.decaySigma ? 2*F/settings.sigmaInitial : 2) < baseF[targetParam][iParam] ? -1 : 1 );
                }
                if ( settings.constraints[iParam] == 0 ) {
                    if ( p[p_err[i].idx] < p.min )
                        p[p_err[i].idx] = p.min;
                    if ( p[p_err[i].idx] > p.max )
                        p[p_err[i].idx] = p.max;
                } else {
                    if ( p[p_err[i].idx] < settings.min[iParam] )
                        p[p_err[i].idx] = settings.min[iParam];
                    if ( p[p_err[i].idx] > settings.max[iParam] )
                        p[p_err[i].idx] = settings.max[iParam];
                }
            } else {
                // Copy non-target params
                p[p_err[i].idx] = p[p_err[source].idx];
            }
        }
    }

    // Reinit
    for ( size_t source = 0; source < settings.nReinit; source++ ) {
        size_t i = p_err.size() - source - 1;
        for ( size_t iParam = 0; iParam < lib.adjustableParams.size(); iParam++ ) {
            if ( settings.constraints[iParam] >= 2 )
                continue;
            AdjustableParam &p = lib.adjustableParams[iParam];
            if ( settings.mutationSelectivity < 2 || iParam == targetParam ) {
                if ( settings.constraints[iParam] == 0 )
                    p[p_err[i].idx] = session.RNG.uniform(p.min, p.max);
                else
                    p[p_err[i].idx] = session.RNG.uniform(settings.min[iParam], settings.max[iParam]);
            } else {
                p[p_err[i].idx] = p[p_err[source].idx];
            }
        }
    }
}

void GAFitter::procreateDE()
{
    std::vector<AdjustableParam> &P = lib.adjustableParams;
    int nParams = P.size();
    int nPop = lib.NMODELS/2;

    std::vector<std::vector<double>> pXList(3);
    std::vector<double> pXmed(3, 0.5);
    std::vector<double> methodCutoff(4);

    // Select the winners
    double bestErr = lib.summary[0];
    int bestIdx = 0;
    for ( int i = 0; i < nPop; i++ ) {
        int iOffspring = i + nPop;
        double err = lib.summary[i];
        if ( lib.summary[i] > lib.summary[iOffspring] ) {
            err = lib.summary[iOffspring];
            for ( int j = 0; j < nParams; j++ ) // replace parent with more successful offspring
                P[j][i] = P[j][iOffspring];
            ++DEMethodSuccess[DEMethodUsed[i]];
        } else {
            for ( int j = 0; j < nParams; j++ ) // replace failed offspring with parent, ready for mutation
                P[j][iOffspring] = P[j][i];
            ++DEMethodFailed[DEMethodUsed[i]];
            if ( DEMethodUsed[i] < 3 )
                pXList[DEMethodUsed[i]].push_back(DEpX[i]);
        }

        if ( err < bestErr ) {
            bestIdx = i;
            bestErr = err;
        }
    }

    // Populate output
    for ( int i = 0; i < nParams; i++ )
        output.params[epoch][i] = P[i][bestIdx];
    output.error[epoch] = std::sqrt(bestErr/errNorm[targetParam]);
    output.targetParam[epoch] = targetParam;

    // Select new target
    targetParam = findNextStim();

    // Calculate mutation probabilities
    double successRateTotal = 0;
    if ( epoch == 0 ) {
        for ( int i = 0; i < 4; i++ ) {
            successRateTotal += 0.25;
            methodCutoff[i] = successRateTotal;
        }
    } else {
        for ( int i = 0; i < 4; i++ ) {
            double successRate = DEMethodSuccess[i] / (DEMethodSuccess[i] + DEMethodFailed[i]) + 0.01;
            successRateTotal += successRate;
            methodCutoff[i] = successRateTotal;

            if ( i < 3 ) {
                auto nth = pXList[i].begin() + pXList[i].size()/2;
                std::nth_element(pXList[i].begin(), nth, pXList[i].end());
                if ( pXList[i].size() % 2 )
                    pXmed[i] = *nth;
                else
                    pXmed[i] = (*nth + *std::max_element(pXList[i].begin(), nth))/2;
                pXList[i].clear();
            }
        }
    }

    // Procreate
    for ( int i = 0; i < nPop; i++ ) {
        double method = session.RNG.uniform(0., successRateTotal);
        double F = session.RNG.variate<double>(0.5, 0.3);
        int r1, r2, r3, r4, r5, forcedJ;
        do { r1 = session.RNG.uniform<int>(0, nPop-1); } while ( r1 == i );
        do { r2 = session.RNG.uniform<int>(0, nPop-1); } while ( r2 == i || r2 == r1 );
        do { r3 = session.RNG.uniform<int>(0, nPop-1); } while ( r3 == i || r3 == r1 || r3 == r2 );
        do { forcedJ = session.RNG.uniform<int>(0, nParams-1); } while ( settings.constraints[forcedJ] >= 2 );

        if ( method < methodCutoff[0] ) {
        // rand/1/bin
            DEMethodUsed[i] = 0;
            DEpX[i] = session.RNG.variate<double>(pXmed[0], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][r1] + F * baseF[targetParam][j] * (P[j][r2] - P[j][r3]);
            }
        } else if ( method < methodCutoff[1] ) {
        // rand-to-best/2/bin
            DEMethodUsed[i] = 1;
            do { r4 = session.RNG.uniform<int>(0, nPop-1); } while ( r4 == i || r4 == r1 || r4 == r2 || r4 == r3 );
            DEpX[i] = session.RNG.variate<double>(pXmed[1], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][i] + F * baseF[targetParam][j] * (P[j][bestIdx] - P[j][i] + P[j][r1] - P[j][r2] + P[j][r3] - P[j][r4]);
            }
        } else if ( method < methodCutoff[2] ) {
        // rand/2/bin
            DEMethodUsed[i] = 2;
            do { r4 = session.RNG.uniform<int>(0, nPop-1); } while ( r4 == i || r4 == r1 || r4 == r2 || r4 == r3 );
            do { r5 = session.RNG.uniform<int>(0, nPop-1); } while ( r5 == i || r5 == r1 || r5 == r2 || r5 == r3 || r5 == r4 );
            DEpX[i] = session.RNG.variate<double>(pXmed[2], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][r1] + F * baseF[targetParam][j] * (P[j][r2] - P[j][r3] + P[j][r4] - P[j][r5]);
            }
        } else {
        // current-to-rand/1
            DEMethodUsed[i] = 3;
            double K = session.RNG.uniform(0.,1.);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 )
                    P[j][i + nPop] = P[j][i] + baseF[targetParam][j] * K * ((P[j][r1] - P[j][i]) + F*(P[j][r2] - P[j][r3]));
            }
        }

        // Apply limits
        for ( int j = 0; j < nParams; j++ ) {
            if ( settings.constraints[j] == 0 ) {
                if ( P[j][i + nPop] < P[j].min )
                    P[j][i + nPop] = P[j].min;
                if ( P[j][i + nPop] > P[j].max )
                    P[j][i + nPop] = P[j].max;
            } else {
                if ( P[j][i + nPop] < settings.min[j] )
                    P[j][i + nPop] = settings.min[j];
                if ( P[j][i + nPop] > settings.max[j] )
                    P[j][i + nPop] = settings.max[j];
            }
        }
    }
}
