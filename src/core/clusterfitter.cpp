#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "clustering.h"

std::vector<std::vector<std::vector<Section>>> GAFitter::constructClustersByStim()
{
    double dt = session.runData().dt;
    int nParams = lib.adjustableParams.size();
    std::vector<double> norm(nParams, 1);

    std::vector<std::vector<std::vector<Section>>> clusters;
    for ( int i = 0; i < nParams; i++ ) {
        iStimulation stim(astims[i], dt);
        session.wavegen().diagnose(stim, dt);
        clusters.push_back(constructClusters(
                                     stim, session.wavegen().lib.diagDelta, settings.cluster_blank_after_step/dt,
                                     nParams+1, norm, settings.cluster_fragment_dur/dt, settings.cluster_threshold,
                                     settings.cluster_min_dur/settings.cluster_fragment_dur));
    }

    return clusters;
}

void GAFitter::resetDE()
{
    DEMethodUsed.assign(session.project.expNumCandidates()/2, 0);
    DEMethodSuccess.assign(4, 0);
    DEMethodFailed.assign(4, 0);
    DEpX.assign(session.project.expNumCandidates()/2, 0);
}

void GAFitter::procreateDE(int errNorm)
{
    std::vector<AdjustableParam> &P = lib.adjustableParams;
    int nParams = P.size();
    int nPop = session.project.expNumCandidates()/2;

    std::vector<std::vector<double>> pXList(3);
    std::vector<double> pXmed(3, 0.5);
    std::vector<double> methodCutoff(4);

    // Select the winners
    double bestErr = lib.err[0];
    int bestIdx = 0;
    for ( int i = 0; i < nPop; i++ ) {
        int iOffspring = i + nPop;
        double err = lib.err[i];
        if ( lib.err[i] > lib.err[iOffspring] ) {
            err = lib.err[iOffspring];
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

        lib.err[i] = lib.err[iOffspring] = 0;
    }

    // Populate output
    for ( int i = 0; i < nParams; i++ )
        output.params[epoch][i] = P[i][bestIdx];
    output.error[epoch] = std::sqrt(bestErr/errNorm);
    output.stimIdx[epoch] = stimIdx;

    // Select new target
    stimIdx = findNextStim();

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
                    P[j][i + nPop] = P[j][r1] + F * baseF[stimIdx][j] * (P[j][r2] - P[j][r3]);
            }
        } else if ( method < methodCutoff[1] ) {
        // rand-to-best/2/bin
            DEMethodUsed[i] = 1;
            do { r4 = session.RNG.uniform<int>(0, nPop-1); } while ( r4 == i || r4 == r1 || r4 == r2 || r4 == r3 );
            DEpX[i] = session.RNG.variate<double>(pXmed[1], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][i] + F * baseF[stimIdx][j] * (P[j][bestIdx] - P[j][i] + P[j][r1] - P[j][r2] + P[j][r3] - P[j][r4]);
            }
        } else if ( method < methodCutoff[2] ) {
        // rand/2/bin
            DEMethodUsed[i] = 2;
            do { r4 = session.RNG.uniform<int>(0, nPop-1); } while ( r4 == i || r4 == r1 || r4 == r2 || r4 == r3 );
            do { r5 = session.RNG.uniform<int>(0, nPop-1); } while ( r5 == i || r5 == r1 || r5 == r2 || r5 == r3 || r5 == r4 );
            DEpX[i] = session.RNG.variate<double>(pXmed[2], 0.1);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 && ( j == forcedJ || session.RNG.uniform(0.,1.) <= DEpX[i] ) )
                    P[j][i + nPop] = P[j][r1] + F * baseF[stimIdx][j] * (P[j][r2] - P[j][r3] + P[j][r4] - P[j][r5]);
            }
        } else {
        // current-to-rand/1
            DEMethodUsed[i] = 3;
            double K = session.RNG.uniform(0.,1.);
            for ( int j = 0; j < nParams; j++ ) {
                if ( settings.constraints[j] < 2 )
                    P[j][i + nPop] = P[j][i] + baseF[stimIdx][j] * K * ((P[j][r1] - P[j][i]) + F*(P[j][r2] - P[j][r3]));
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

double GAFitter::clusterDE()
{
    std::vector<std::tuple<int, std::vector<double>, std::vector<Section>>> options;
    //                     stim, F,  cluster
    options = extractSeparatingClusters(constructClustersByStim(), lib.adjustableParams.size());

    populate();
    resetDE();

    std::vector<int> errNorm(lib.adjustableParams.size(), 0);
    baseF.clear();
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        baseF.push_back(std::get<1>(options[i]));
        for ( Section &sec : std::get<2>(options[i]) )
            errNorm[i] += sec.end - sec.start;
    }

    double simtime = 0;
    for ( epoch = 0; !finished(); ++epoch ) {
        simtime += stimulate_cluster(std::get<2>(options[stimIdx]), std::get<0>(options[stimIdx]));

        lib.pullErr();
        procreateDE(errNorm[stimIdx]);
        lib.push();

        emit progress(epoch);
    }

    return simtime;
}

double GAFitter::clusterGA()
{
    std::vector<AdjustableParam> &P = lib.adjustableParams;
    int nParams = P.size();

    std::vector<std::tuple<int, std::vector<double>, std::vector<Section>>> options;
    options = extractSeparatingClusters(constructClustersByStim(), nParams);

    populate();

    double simtime = 0;
    for ( epoch = 0; !finished(); epoch++ ) {
        // Stimulate
        simtime += stimulate_cluster(std::get<2>(options[stimIdx]), std::get<0>(options[stimIdx]));

        // Advance
        lib.pullErr();
        procreate();
        lib.push();

        emit progress(epoch);
    }

    return simtime;
}

double GAFitter::windowedDE()
{
    populate();
    resetDE();

    int nParams = lib.adjustableParams.size();
    baseF.assign(nParams, std::vector<double>(nParams, 0));
    for ( int i = 0; i < nParams; i++ ) {
        baseF[i][i] = 1;
    }

    double simtime = 0;
    for ( epoch = 0; !finished(); ++epoch ) {
        simtime += stimulate();

        lib.pullErr();
        procreateDE(stims[stimIdx].tObsEnd - stims[stimIdx].tObsBegin);
        lib.push();

        emit progress(epoch);
    }

    return simtime;
}

double GAFitter::stimulate_cluster(const std::vector<Section> &cluster, int stimIdx)
{
    const iStimulation &I = stims.at(stimIdx);
    const Stimulation &aI = astims.at(stimIdx);
    const RunData &rd = session.runData();
    int iTStep;

    // Set up library
    lib.t = 0.;
    lib.iT = 0;
    lib.VC = true;

    // Settle library
    lib.getErr = false;
    lib.getLikelihood = false;
    lib.setVariance = false;
    lib.VClamp0 = I.baseV;
    lib.dVClamp = 0;
    lib.step(rd.settleDuration, rd.simCycles * int(rd.settleDuration/rd.dt), true);

    // Set up + settle DAQ
    daq->VC = true;
    daq->reset();
    daq->run(aI, rd.settleDuration);
    for ( size_t iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
        daq->next();

    lib.t = 0;
    lib.iT = 0;

    auto secIter = cluster.begin();
    while ( (int)lib.iT < I.duration ) {
        daq->next();
        lib.Imem = daq->current;

        // Populate VClamp0/dVClamp with the next "segment" of length tSpan = iTStep = 1
        getiCommandSegment(I, lib.iT, 1, rd.dt, lib.VClamp0, lib.dVClamp, iTStep);

        pushToQ(qT + lib.t, daq->voltage, daq->current, lib.VClamp0+lib.t*lib.dVClamp);

        lib.getErr = secIter != cluster.end() && lib.iT >= size_t(secIter->start);

        lib.step();

        if ( lib.getErr && lib.iT == size_t(secIter->end) ) {
            ++secIter;
        }
    }

    qT += aI.duration;
    daq->reset();

    return astims.at(stimIdx).duration + session.runData().settleDuration;
}
