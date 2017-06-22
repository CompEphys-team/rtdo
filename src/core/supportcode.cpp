#ifndef SUPPORTCODE_CU
#define SUPPORTCODE_CU

#include "supportcode.h"

__device__ void closeBubble(WaveStats &s, const scalar t)
{
    if ( s.current.cycles ) {
        scalar fitness = s.current.rel / s.current.cycles;
        if ( fitness > s.fitness ) {
            s.best = {
                s.current.cycles,
                t
            };
            s.fitness = fitness;
        }
        s.current = {};
        s.bubbles++;
    }
}

__device__ void processStats(const scalar err,  //!< Target parameter's absolute deviation from base model on this cycle
                             const scalar mean, //!< Mean deviation across all parameters
                             const scalar t,    //!< Time, including substep contribution
                             WaveStats &s) //!< Stats struct for this group & target param
{
    scalar rel = err/mean;
    if ( rel > 1 ) {
        s.current.rel += rel;
        s.current.cycles++;
    } else if ( s.current.cycles ) {
        closeBubble(s, t);
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
    if ( s->t > t ) {
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
