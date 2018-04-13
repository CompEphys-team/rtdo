#ifndef SUPPORTCODE_H
#define SUPPORTCODE_H

#include <cuda_runtime.h>
#include "types.h"

__host__ __device__ scalar getCommandVoltage(const Stimulation &I, scalar t);

//! Calculates command voltage extrapolated to t=0 and the corresponding gradient (mV/ms) of the current and potentially next step.
//! Returns the step time and populates the second pair of values if a step occurs within [t, t+dt]; 0 otherwise.
scalar getCommandVoltages(const Stimulation &I, scalar t, scalar dt,
                          scalar &VClamp0, scalar &dVClamp,
                          scalar &VClamp0_2, scalar &dVClamp_2);

//! Calculates command voltage extrapolated to t=0, the corresponding gradient (mV/ms), and the duration of the first linear segment in [t, t+dt]
//! Returns a flag indicating whether additional steps are required, i.e., whether there are multiple linear segments in [t, t+dt]
//! @a res defines the output resolution, i.e. the time between command updates
//! @a res_t0 defines the time of first command output, provided to allow unaligned output (as in comedidaq)
__host__ __device__ bool getCommandSegment(const Stimulation &I, scalar t, scalar dt, scalar res, scalar res_t0,
                       scalar &VClamp0, scalar &dVClamp, scalar &tStep);

struct ClampParameters {
    scalar clampGain, accessResistance;
    scalar VClamp0; // = VClamp linearly extrapolated to t=0
    scalar dVClamp; // = VClamp gradient, mV/ms

    __host__ __device__ inline scalar getCurrent(scalar t, scalar V) const
    {
        return (clampGain * (VClamp0 + t*dVClamp - V) - V) / accessResistance;
    }
};

namespace RKF45_Constants {
    constexpr scalar c2 = 1./4.;
    constexpr scalar c3 = 3./8.;
    constexpr scalar c4 = 12./13.;
    constexpr scalar c5 = 1.;
    constexpr scalar c6 = 1./2.;

    constexpr scalar a21 = 1./4.;
    constexpr scalar a31 = 3./32.;
    constexpr scalar a32 = 9./32.;
    constexpr scalar a41 = 1932./2197.;
    constexpr scalar a42 = -7200./2197.;
    constexpr scalar a43 = 7296./2197.;
    constexpr scalar a51 = 439./216.;
    constexpr scalar a52 = -8;
    constexpr scalar a53 = 3680./513.;
    constexpr scalar a54 = -845./4104.;
    constexpr scalar a61 = -8./27.;
    constexpr scalar a62 = 2;
    constexpr scalar a63 = -3544./2565.;
    constexpr scalar a64 = 1859./4104.;
    constexpr scalar a65 = -11./40.;

    constexpr scalar b1 = 16./135.;
    constexpr scalar b3 = 5565./12825.;
    constexpr scalar b4 = 28561./56430.;
    constexpr scalar b5 = -9./50.;
    constexpr scalar b6 = 2./55.;

    // Fourth-order estimate parameters are inverted around 0 for efficiency (avoid one call to State::operator*(-1) per iteration)
    constexpr scalar b_1 = -25./216.;
    constexpr scalar b_3 = -1408./2565.;
    constexpr scalar b_4 = -2197./4104.;
    constexpr scalar b_5 = 1./5.;
}

template <class StateT, class ParamsT>
__host__ __device__ int RKF45(scalar t, scalar tEnd, scalar hMin, scalar hMax, scalar &hP,
                               StateT &state, const ParamsT &params, const ClampParameters &clamp)
{
    using namespace RKF45_Constants;
    scalar h = hP, delta = 1;
    int n = 0;
    bool success;
    while ( t < tEnd ) {
        h *= delta;
        hP = h = scalarmin(hMax, scalarmax(hMin, h));
        if ( h >= tEnd-t )
            h = tEnd-t; // Truncate final step to return a precise value at t = tEnd

        StateT k1 = state.state__f(t, params, clamp);
        StateT k2 = StateT(state + k1*(a21*h))                                                    .state__f(t + h*c2, params, clamp);
        StateT k3 = StateT(state + k1*(a31*h) + k2*(a32*h))                                       .state__f(t + h*c3, params, clamp);
        StateT k4 = StateT(state + k1*(a41*h) + k2*(a42*h) + k3*(a43*h))                          .state__f(t + h*c4, params, clamp);
        StateT k5 = StateT(state + k1*(a51*h) + k2*(a52*h) + k3*(a53*h) + k4*(a54*h))             .state__f(t + h*c5, params, clamp);
        StateT k6 = StateT(state + k1*(a61*h) + k2*(a62*h) + k3*(a63*h) + k4*(a64*h) + k5*(a65*h)).state__f(t + h*c6, params, clamp);

        StateT dEst5 = k1*(b1*h) + k3*(b3*h) + k4*(b4*h) + k5*(b5*h) + k6*(b6*h);
        delta = StateT(dEst5
                       + k1*(b_1*h) + k3*(b_3*h) + k4*(b_4*h) + k5*(b_5*h) // == -dEst4
                       ).state__delta(h, success);

        if ( success || h <= hMin ) {
            t += h;
            state = state + dEst5;
            state.state__limit();
        }
        ++n;
    }
    return n;
}

template <class StateT, class ParamsT>
__host__ __device__ inline void RK4(scalar t, scalar h,
                                    StateT &state, const ParamsT &params, const ClampParameters &clamp,
                                    scalar noiseI[3] = 0)
{
    StateT k1 = state.state__f(t, params, clamp.getCurrent(t, state.V) + (noiseI ? noiseI[0] : 0));
    StateT est = state + k1*(h/2);

    StateT k2 = est.state__f(t+h/2, params, clamp.getCurrent(t+h/2, est.V) + (noiseI ? noiseI[1] : 0));
    est = state + k2*(h/2);

    StateT k3 = est.state__f(t+h/2, params, clamp.getCurrent(t+h/2, est.V) + (noiseI ? noiseI[1] : 0));
    est = state + k3*h;

    StateT k4 = est.state__f(t+h, params, clamp.getCurrent(t+h, est.V) + (noiseI ? noiseI[2] : 0));

    state = state + (k1 + k2*2 + k3*2 + k4) * (h/6);
    state.state__limit();
}

template <class StateT, class ParamsT>
__host__ __device__ inline void Euler(scalar t, scalar h,
                                      StateT &state, const ParamsT &params, const ClampParameters &clamp)
{
    state = state + state.state__f(t, params, clamp) * h;
}

#endif // SUPPORTCODE_H
