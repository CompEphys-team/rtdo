#include "lib_definitions.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

/**
 * @brief compare_traces
 * @param nTraces
 * @param nSamples
 * @param dV_threshold
 * @param dV_decay
 * @param SDF_size
 * @param SDF_decay
 * @param errors
 * 
 * Notes on addressing:
 * - Each stim is evaluated against nTraces models, yielding nTraces voltage traces.
 * - blockIdx.y addresses over stims
 * - Within a stim, we compare nTraces all-to-all, without duplicates
 * - Consider a square of nTraces*nTraces comparisons. Cut it into warpSize*warpSize units. Each unit is served by a single warp.
 * - blockIdx.x addresses a row of such units.
 * - The first served unit in each row is on the diagonal.
 * - Block size should be nTraces/2 + warpSize, to cover the diagonal and half of the right-hand (i.e. above-diagonal) units, wrapping around
 * - This necessarily raises a duplicacy issue; the final warps in the lower half of the trace square (blockIdx.x >= gridDim.x/2) would produce duplicate data and are therefore exited immediately
 * - Each warp, serving a unit of warpSize*warpSize trace comparisons, likewise works in diagonal fashion
 * - The true diagonal (comparing trace k against itself) is not computed
 * - The same duplicacy issue arises within a warp, too, with the final comparison (warpSize/2 to the right of the diagonal, wrapping around) computed by two threads (e.g. 0vs16 in lane 0, and 16vs0 in lane 16)
 *
 * Assuming eight units (nTraces==256), block numbers are as follows:
 * 0 0 0 0 0 x x x
 * x 1 1 1 1 1 x x
 * x x 2 2 2 2 2 x
 * x x x 3 3 3 3 3
 * 4 x x x 4 4 4 4
 * 5 5 x x x 5 5 5
 * 6 6 6 x x x 6 6
 * 7 7 7 7 x x x 7
 *
 * Note how the final 4-7 duplicate the final 0-3.
 * The same organisational principle is at play within each warp: Consider the numbers above as laneids; the "compare" loop below runs through each instance from the diagonal rightward.
 * Saving is done starting from the diagonal also, then moving on to the first above-diagonal, etc.
 *
 * Note, an n-by-n square of comparisons produces either n(n-1)/2 unique comparisons if the diagonal is excluded (warpid==0), or n(n+1)/2 unique comparisons if the diagonal is included (warpid>0).
 */
__global__ void compare_traces_kernel(int nTraces, int nSamples, scalar dV_threshold, scalar dV_decay, scalar SDF_size, scalar SDF_decay, scalar *errors)
{
    constexpr int min_t_spike = 20;
    
    const int warpid = (threadIdx.x / warpSize), laneid = threadIdx.x % warpSize;
    const int traceOffset1 = blockIdx.y * nTraces + ((blockIdx.x * warpSize + warpid * warpSize) % nTraces) + laneid;
    const int traceOffset2 = blockIdx.y * nTraces + blockIdx.x * warpSize + laneid;

    if ( blockIdx.x >= gridDim.x/2 && warpid == gridDim.x/2 ) // Drop duplicate warps
        return;
    
    scalar V1, V2, oldV1 = 0, oldV2 = 0, dV1 = 0, dV2 = 0, SDF1 = 0, SDF2 = 0;
    scalar diff;
    bool spike1 = false, spike2 = false;
    double summary[17 /* = warpSize/2 + 1*/] = {};
    
    for ( int t = 0; t < nSamples; t++ ) {
        // Get samples
        V1 = dd_timeseries[t*NMODELS + traceOffset1];
        V2 = warpid ? dd_timeseries[t*NMODELS + traceOffset2] : V1;
        
        // Detect spikes
        if ( t > min_t_spike ) {
            if ( !spike1 && dV1 > dV_threshold ) {
                spike1 = true;
                SDF1 += SDF_size;
            } else if ( spike1 && dV1 < 0 ) {
                spike1 = false;
            }
            if ( !spike2 && dV2 > dV_threshold ) {
                spike2 = true;
                SDF2 += SDF_size;
            } else if ( spike2 && dV2 < 0 ) {
                spike2 = false;
            }
        }
        
        // Compare
        for ( int i = warpid ? 0 : 1; i < warpSize/2+1; i++ ) {
            diff = V1 - __shfl_sync(0xffffffff, V2, laneid+i);
            summary[i] += spike1 ? 0 : diff*diff;
            diff = SDF1 - __shfl_sync(0xffffffff, SDF2, laneid+i);
            summary[i] += diff*diff;
        }
        
        // Decay
        SDF1 *= SDF_decay;
        SDF2 *= SDF_decay;
        dV1 = dV1*dV_decay + (V1 - oldV1);
        dV2 = dV2*dV_decay + (V2 - oldV2);
        oldV1 = V1;
        oldV2 = V2;
    }

    int last = warpSize/2 + (warpid < warpSize/2 ? 1 : 0); // Second half-warp must not save its final summary (duplicate)
    for ( int i = warpid ? 0 : 1; i < last; i++ ) {
        summary[i] /= nSamples;
        summary[0] += i ? summary[i] : 0; // Add all errors within thread
    }

    __syncwarp();
    summary[0] = warpReduceSum(summary[0]);
    if ( laneid == 0 )
        atomicAdd(&errors[blockIdx.y], (scalar)summary[0]);
}

extern "C" std::vector<scalar> cl_get_mean_cost(int nStims, int nSamples, scalar dV_threshold, scalar dV_decay, scalar SDF_size, scalar SDF_decay)
{
    int nTraces = NMODELS / nStims;
    int nUnits = nTraces / 32;
    int nComparisonsPerStim = nTraces*(nTraces-1)/2;
    std::vector<scalar> means(nStims, 0);

    CHECK_CUDA_ERRORS(cudaMemcpy(d_prof_error, means.data(), nStims*sizeof(scalar), cudaMemcpyHostToDevice));

    dim3 grid(nUnits, nStims);
    dim3 block(nTraces/2 + 32);
    compare_traces_kernel<<<grid, block>>>(nTraces, nSamples, dV_threshold, dV_decay, SDF_size, SDF_decay, d_prof_error);

    CHECK_CUDA_ERRORS(cudaMemcpy(means.data(), d_prof_error, nStims*sizeof(scalar), cudaMemcpyDeviceToHost));

    for ( scalar &m : means )
        m /= nComparisonsPerStim;
    return means;
}
