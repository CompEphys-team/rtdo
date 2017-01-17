#ifndef SUPPORTCODE_CU
#define SUPPORTCODE_CU

#include "supportcode.h"

__device__ void processStats(const scalar err,  //!< Target parameter's absolute deviation from base model on this cycle
                             const scalar mean, //!< Mean deviation across all parameters
                             const scalar t,    //!< Time, including substep contribution
                             WaveStats &s, //!< Stats struct for this group & target param
                             const bool final)  //!< True if this is the very last cycle (= force close bubbles)
{
    scalar rel = err/mean;
    if ( !final && rel > 1 ) {
        s.current.rel += rel;
        s.current.cycles++;
    } else if ( s.current.cycles ) {
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
