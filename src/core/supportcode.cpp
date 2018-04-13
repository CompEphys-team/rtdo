#ifndef SUPPORTCODE_CU
#define SUPPORTCODE_CU

#include "supportcode.h"

__device__ inline scalar fitnessPartial(scalar err, scalar mean)
{
    return err/mean;
}

__device__ void closeBubble(Bubble &best, Bubble &current, int mt)
{
    if ( current.startCycle >= 0 ) {
        int cycles = mt - current.startCycle;
        if ( !best.cycles || current.value / cycles > best.value / best.cycles ) {
            best.startCycle = current.startCycle;
            best.cycles = cycles;
            best.value = current.value;
        }
        current.startCycle = -1;
        current.value = 0;
    }
}

__device__ void extendBubble(Bubble &current, scalar parfit, int mt)
{
    current.value += parfit;
    if ( current.startCycle < 0 )
        current.startCycle = mt;
}

// Returns the command voltage; populates @a step with V_(offset per dt, if ramp is true), t_(next step), and ramp_(now).
__device__ scalar getStep(Stimulation::Step &step, const Stimulation &I, scalar t, scalar dt)
{
    if ( I.empty() ) {
        step.t = I.duration;
        step.ramp = false;
        return I.baseV;
    }
    const Stimulation::Step *s = I.begin();
    if ( s->t > t ) {
        step.t = s->t;
        step.V = s->ramp ? (s->V - I.baseV) * dt/s->t : 0;
        step.ramp = s->ramp;
        return I.baseV + t*step.V;
    } else {
        while ( s != I.end() && s->t <= t )
            s++;
        if ( s != I.end() ) {
            step.t = s->t;
            step.V = s->ramp ? (s->V - (s-1)->V) * dt/(s->t - (s-1)->t) : 0;
            step.ramp = s->ramp;
        } else {
            step.t = I.duration;
            step.ramp = false;
        }
        return (s-1)->V;
    }
}

scalar getCommandVoltages(const Stimulation &I, scalar t, scalar dt,
                        scalar &VClamp0, scalar &dVClamp,
                        scalar &VClamp0_2, scalar &dVClamp_2)
{
    if ( I.empty() || t < 0 || t >= I.duration ) {
        dVClamp = 0;
        VClamp0 = I.baseV;
        return 0;
    }
    const Stimulation::Step *s = I.begin();

    // First partial interval
    if ( t < s->t ) {
        dVClamp = s->ramp ? (s->V - I.baseV) / s->t : 0;
        VClamp0 = I.baseV;
    } else {
        while ( s != I.end() && s->t <= t )
            s++;
        if ( s == I.end() ) {
            dVClamp = 0;
            VClamp0 = (s-1)->V;
            return 0;
        } else {
            dVClamp = s->ramp ? (s->V - (s-1)->V) / (s->t - (s-1)->t) : 0;
            VClamp0 = s->ramp ? s->V - dVClamp*s->t : (s-1)->V;
        }
    }

    // Second partial interval
    if ( t + dt > s->t ) {
        if ( s+1 == I.end() ) {
            dVClamp_2 = 0;
            VClamp0_2 = s->V;
        } else {
            dVClamp_2 = (s+1)->ramp ? ((s+1)->V - s->V) / ((s+1)->t - s->t) : 0;
            VClamp0_2 = (s+1)->ramp ? (s+1)->V - dVClamp_2*(s+1)->t : s->V;
        }
        return s->t;
    } else {
        return 0;
    }
}

__host__ __device__ bool getCommandSegment(const Stimulation &I, scalar t, scalar dt, scalar res, scalar res_t0,
                       scalar &VClamp0, scalar &dVClamp, scalar &tStep)
{
    if ( I.empty() || t < 0 || t >= I.duration ) {
        dVClamp = 0;
        VClamp0 = I.baseV;
        tStep = dt;
        return false;
    }
    const Stimulation::Step *s = I.begin();

    // Adjust t to resolution, rounding down (adding a safety margin against numeric inaccuracy where res neatly divides t)
    // Note that in most cases this does not change t; it only affects cases where res is unaligned to DT (i.e. ComediDAQ)
    if ( res > 0 )
        t = res_t0 + res * int((t - res_t0) / res + 1e-5);

    // Find segment at t
    if ( t < s->t ) {
        dVClamp = s->ramp ? (s->V - I.baseV) / s->t : 0;
        VClamp0 = I.baseV;
    } else {
        while ( s != I.end() && s->t <= t )
            s++;
        if ( s == I.end() ) {
            // Final segment, definitely no further chunking
            dVClamp = 0;
            VClamp0 = (s-1)->V;
            tStep = dt;
            return false;
        } else {
            dVClamp = s->ramp ? (s->V - (s-1)->V) / (s->t - (s-1)->t) : 0;
            VClamp0 = s->ramp ? s->V - dVClamp*s->t : (s-1)->V;
        }
    }

    if ( res > 0 ) // Divide time to step by res, rounding up (no safety margin: s->t is unaligned anyway)
        tStep = res_t0 + res * std::ceil((s->t - res_t0) / res) - t;
    else
        tStep = s->t - t;

    if ( tStep < dt ) // If the next step is within the interval, split it
        return true;

    tStep = dt;
    return false;
}

__host__ __device__ scalar getCommandVoltage(const Stimulation &I, scalar t)
{
    scalar Vcmd;
    if ( I.empty() )
        return I.baseV;
    const Stimulation::Step *s = I.begin();
    if ( t < 0 || t >= I.duration ) {
        Vcmd = I.baseV;
    } else if ( t < s->t ) {
        if ( s->ramp )
            Vcmd = I.baseV + (s->V - I.baseV) * t/s->t;
        else
            Vcmd = I.baseV;
    } else {
        while ( s != I.end() && s->t <= t )
            s++;
        if ( s != I.end() && s->ramp )
            Vcmd = (s-1)->V + (s->V - (s-1)->V) * (t - (s-1)->t) / (s->t - (s-1)->t);
        else
            Vcmd = (s-1)->V;
    }
    return Vcmd;
}

#endif // SUPPORTCODE_CU
