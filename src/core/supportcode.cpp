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

__device__ void extendBubble(Bubble &best, Bubble &current, scalar parfit, int mt)
{
    if ( parfit > 1 ) {
        current.value += parfit;
        if ( current.startCycle < 0 )
            current.startCycle = mt;
    } else {
        closeBubble(best, current, mt);
    }
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

__host__ __device__ scalar getCommandVoltage(const Stimulation &I, scalar t)
{
    scalar Vcmd;
    if ( I.empty() )
        return I.baseV;
    const Stimulation::Step *s = I.begin();
    if ( t < s->t ) {
        if ( s->ramp )
            Vcmd = I.baseV + (s->V - I.baseV) * t/s->t;
        else
            Vcmd = I.baseV;
    } else if ( t >= I.duration ) {
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
