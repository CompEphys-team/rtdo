#include "gafitter.h"
#include "session.h"
#include "supportcode.h"
#include "clustering.h"

double GAFitter::fit()
{
    double simtime = 0;
    for ( epoch = 0; !finished(); epoch++ ) {
        // Stimulate
        simtime += stimulate();

        // Advance
        lib.pullSummary();
        if ( settings.useDE )
            procreateDE();
        else
            procreate();

        emit progress(epoch);
    }

    return simtime;
}

quint32 GAFitter::findNextStim()
{
    // Exclude stims with fixed parameter value (constraints==2)
    // translate stimIdx to point to a contiguous array of the actually used stimulations
    quint32 nStims = 0, previousStimIdx = 0;
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        if ( settings.constraints[i] < 2 ) {
            ++nStims;
            if ( i < targetParam )
                ++previousStimIdx;
        }
    }

    quint32 nextStimIdx(previousStimIdx);
    if ( settings.randomOrder == 3 ) { // Sequence-biased random
        // bias[i] == number of epochs since i was last used
        if ( epoch == 0 ) {
            for ( size_t i = 0; i < nStims; i++ )
                bias[i] = 1;
        } else {
            bias[previousStimIdx] = 0;
            for ( size_t i = 0; i < nStims; i++ )
                ++bias[i];
        }
        std::vector<int> cumBias(nStims, bias[0]);
        for ( size_t i = 1; i < nStims; i++ ) // Cumulative sum
            cumBias[i] = cumBias[i-1] + bias[i];
        int choice = session.RNG.uniform<int>(0, cumBias.back()-1);
        for ( nextStimIdx = 0; choice >= cumBias[nextStimIdx]; nextStimIdx++ ) ;
    } else if ( settings.randomOrder == 2 ) { // Error-biased random
        double cost = output.error[epoch];
        if ( epoch == previousStimIdx ) // Initial: Full error
            bias[previousStimIdx] = cost;
        else // Recursively decay bias according to settings
            bias[previousStimIdx] = settings.orderBiasDecay * cost + (1-settings.orderBiasDecay) * bias[previousStimIdx];

        if ( epoch + 1 < nStims ) { // Initial round: Sequential order
            nextStimIdx = previousStimIdx + 1;
        } else if ( int(epoch) < settings.orderBiasStartEpoch ) { // Further unbiased rounds: Random order
            nextStimIdx = session.RNG.uniform<quint32>(0, nStims-1);
        } else { // Biased rounds
            double sumBias = 0;
            for ( size_t i = 0; i < nStims; i++ )
                sumBias += bias[i];
            double choice = session.RNG.uniform(0.0, sumBias);
            for ( size_t i = 0; i < nStims; i++ ) {
                choice -= bias[i];
                if ( choice < 0 ) {
                    nextStimIdx = i;
                    break;
                }
            }
        }
    } else if ( settings.randomOrder == 1 )
        nextStimIdx = session.RNG.uniform<quint32>(0, nStims-1);
    else
        nextStimIdx = (previousStimIdx + 1) % nStims;

    // Translate nextStimIdx back to index into full stim array
    for ( size_t i = 0; i <= nextStimIdx; i++ )
        if ( settings.constraints[i] >= 2 )
            ++nextStimIdx;

    return nextStimIdx;
}

double GAFitter::finalise()
{
    std::vector<errTupel> f_err(lib.NMODELS);
    int t = 0;
    double dt = session.runData().dt, simt = 0;

    // restore original stims as specified in source
    astims = output.stimSource.stimulations();
    stims = output.stimSource.iStimulations(dt);

    // Evaluate existing population on all stims
    for ( targetParam = 0; targetParam < stims.size() && !isAborted(); targetParam++ ) {
        obs[targetParam] = iObserveNoSteps(stims[targetParam], session.wavegenData().cluster.blank/dt);
        t += obs[targetParam].duration();
        simt += stimulate(targetParam>0 ? ASSIGNMENT_SUMMARY_PERSIST : 0);
    }

    // Pull & sort by total cumulative error across all stims
    lib.pullSummary();
    for ( size_t i = 0; i < lib.NMODELS; i++ ) {
        f_err[i].idx = i;
        f_err[i].err = lib.summary[i];
    }
    auto winner = f_err.begin();
    std::nth_element(f_err.begin(), winner, f_err.end(), &errTupelSort);

    double err = std::sqrt(winner->err / t);
    for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
        output.finalParams[i] = lib.adjustableParams[i][winner->idx];
        output.finalError[i] = err;
    }
    output.final = true;

    return simt;
}

double GAFitter::stimulate(unsigned int extra_assignments)
{
    const RunData &rd = session.runData();
    const Stimulation &aI = astims[targetParam];
    iStimulation I = stims[targetParam];

    // Set up library
    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.setRundata(0, rd);

    lib.setSingularStim();
    lib.stim[0] = I;
    lib.obs[0] = obs[targetParam];

    lib.setSingularTarget();
    lib.resizeTarget(1, I.duration);
    lib.targetOffset[0] = 0;

    lib.assignment = lib.assignment_base | extra_assignments
            | ASSIGNMENT_REPORT_SUMMARY | ASSIGNMENT_SUMMARY_COMPARE_TARGET | ASSIGNMENT_SUMMARY_SQUARED;
    if ( !rd.VC )
        lib.assignment |= ASSIGNMENT_PATTERNCLAMP | ASSIGNMENT_PC_REPORT_PIN;
    lib.summaryOffset = 0;
    lib.push();

    // Set up + settle DAQ
    daq->reset();
    daq->run(aI, rd.settleDuration);
    if ( rd.VC ) {
        for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ )
            daq->next();
    } else if ( rd.settleDuration > 0 ) {
        for ( int iT = 0, iTEnd = rd.settleDuration/rd.dt; iT < iTEnd; iT++ ) {
            daq->next();
            lib.target[iT] = daq->voltage;
        }
        lib.assignment |= ASSIGNMENT_SETTLE_ONLY;
        lib.pushTarget();
        lib.run();
    }

    // Step DAQ through full stimulation
    for ( int iT = 0; iT < I.duration; iT++ ) {
        daq->next();
        pushToQ(qT + iT*rd.dt, daq->voltage, daq->current, getCommandVoltage(aI, iT*rd.dt));
        lib.target[iT] = rd.VC ? daq->current : daq->voltage;
    }
    daq->reset();

    if ( !rd.VC && rd.settleDuration > 0 ) {
        lib.assignment &= ~ASSIGNMENT_SETTLE_ONLY;
        lib.iSettleDuration[0] = 0;
        lib.push(lib.iSettleDuration);
    }

    // Run lib against target
    lib.pushTarget();
    lib.run();

    qT += I.duration * rd.dt;

    return I.duration * rd.dt + rd.settleDuration;
}