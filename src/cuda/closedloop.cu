#include "lib_definitions.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>


#define CL_PROCESS_KERNSZ 32
#define NSPIKES 32
#define NSPIKES_MASK 0xff
#define SPIKE_DETECTED 0x100
#define PEAK_DETECTED 0x200
#define DELAYSZ 16
#define DMAP_MAXENTRIES 1024
#define DMAP_ENTRY_STRIDE 41 /* Being a factor of 1025, striding by 41 mod 1024 fills the list perfectly while keeping adjacent entries far from each other */
#define DMAP_SIZE 64
#define DMAP_STRIDE 16
#define DMAP_KW 16 /* kernel width */

scalar *h_filtV = nullptr, *d_filtV = nullptr;
unsigned int filtV_size = 0;

int *spiketimes_models = nullptr;
unsigned int spiketimes_size = 0;
scalar *spikemarks_models = nullptr;
unsigned int spikemarks_size = 0;

scalar *d_dmaps = nullptr;
unsigned int dmaps_size = 0;

scalar *d_Vx = nullptr, *d_Vy = nullptr;
__constant__ scalar *Vx = nullptr, *Vy = nullptr;
unsigned int Vx_size = 0, Vy_size = 0;

__constant__ int spiketimes_target[NSPIKES];
__constant__ scalar spikemarks_target[NSPIKES];
__constant__ scalar dmap_target[DMAP_SIZE * DMAP_SIZE];
int h_spiketimes_target[NSPIKES];
scalar h_spikemarks_target[NSPIKES];
scalar h_dmap_target[DMAP_SIZE * DMAP_SIZE];


/// This calculates the cross-terms of the van Rossum distance using the algorithm in Houghton & Kreuz (2012), http://sci-hub.tw/10.3109/0954898X.2012.673048
__device__ scalar cl_get_sdf_crossterms(const int nSpikes, int *t_self, int *t_target, scalar *m_self, scalar *m_target, scalar sdf_tau, int tEnd, int stride = 1)
{
    scalar err = 0;
    for ( int i = 0, j = -1; i < nSpikes; i++ ) {
        while ( j < NSPIKES-1 && t_target[(j+1)*stride] < t_self[i*stride] )
            ++j;
        err += (j == -1) ? 0 : (m_target[j*stride] + 1) * scalarexp((t_target[j*stride] - t_self[i*stride]) / sdf_tau); // Cross-term with m(target)
    }
    for ( int i = -1, j = 0; j < NSPIKES && t_target[j*stride] < tEnd; j++ ) {
        while ( i < nSpikes-1 && t_self[(i+1)*stride] < t_target[j*stride] )
            ++i;
        err += (i == -1) ? 0 : (m_self[i*stride] + 1) * scalarexp((t_self[i*stride] - t_target[j*stride]) / sdf_tau); // Cross-term with m(self)
    }
    return err;
}


/**
 * Notes on addressing:
 * - Each stim is evaluated against nTraces models, yielding nTraces voltage traces.
 * - blockIdx.z addresses over stims
 * - Within a stim, we compare nTraces all-to-all, without duplicates or self-reference, yielding nTraces*(nTraces-1)/2 comparisons.
 * - Consider a square of nTraces*nTraces comparisons. Cut it into warpSize*warpSize units. Each unit is served by a single block with a single 1-D warp.
 * - blockIdx.x/y addresses a unit's column/row.
 * - blocks below the diagonal (y>x) are dropped immediately.
 *
 * - Within each block, threads load data for both (blockIdx.)x and y coordinates.
 * - They then compare internally (except on the true diagonal) and to their right-hand neighbours with offset i=1 to 16, wrapping around.
 * - This necessarily raises a duplicacy issue, where the final comparison is done twice (once in the first half-warp, e.g. lane 0 vs lane 16, and once in the second, lane 16 vs lane 0).
 *     This duplicacy is corrected for appropriately, see the `(i == HALFWARP && threadId.x >= HALFWARP)` conditions.
 */
__global__ void compare_models_kernel(int nTraces, int nSamples, int *st_in, scalar *sm_in, scalar *dmap, scalar sdf_tau, scalar err_weight_trace, scalar err_weight_sdf, scalar err_weight_dmap, scalar *errors)
{
    if ( blockIdx.y > blockIdx.x )
        return; // Do not calculate the duplicate below-diagonal units

    constexpr int HALFWARP = 16;
    const int offset1 = blockIdx.z*nTraces + blockIdx.x*warpSize + threadIdx.x;
    const int offset2 = blockIdx.z*nTraces + blockIdx.y*warpSize + threadIdx.x;
    const bool ndiag = (offset1 != offset2);

    scalar err;
    double err_trace = 0;

    __shared__ int t1[2*HALFWARP*NSPIKES], t2[2*HALFWARP*NSPIKES];
    __shared__ scalar m1[2*HALFWARP*NSPIKES], m2[2*HALFWARP*NSPIKES];
    int nSpikes = 0;
    scalar sum_m_N = 0;
    double err_sdf = 0;

    double err_dmaps[HALFWARP+1] = {};

    // Compare voltage traces
    for ( int t = 0; t < nSamples; t++ ) {
        // Get samples
        scalar V1 = dd_timeseries[t*NMODELS + offset1];
        scalar V2 = ndiag ? dd_timeseries[t*NMODELS + offset2] : V1;

        // Compare
        for ( int i = ndiag ? 0 : 1; i < HALFWARP+1; i++ ) {
            err = V1 - __shfl_sync(0xffffffff, V2, threadIdx.x+i);
            err_trace += (i == HALFWARP && threadIdx.x >= HALFWARP) ? 0 : scalarfabs(err);
        }
    }

    // Compare SDF
    // load
    for ( int i = 0; i < NSPIKES; i++ ) {
        int t = st_in[i*NMODELS + offset1];
        scalar m = sm_in[i*NMODELS + offset1];
        int tt = ndiag ? st_in[i*NMODELS + offset2] : t;
        int mm = ndiag ? sm_in[i*NMODELS + offset2] : m;
        t1[i*warpSize + threadIdx.x] = t;
        m1[i*warpSize + threadIdx.x] = m;
        t2[i*warpSize + threadIdx.x] = tt;
        m2[i*warpSize + threadIdx.x] = mm;
        if ( t < nSamples ) {
            ++nSpikes;
            sum_m_N += m + 0.5;
        }
        if ( t > nSamples && tt > nSamples )
            break;
    }
    __syncwarp();
    // compare
    for ( int i = ndiag ? 0 : 1; i < HALFWARP+1; i++ ) {
        err = cl_get_sdf_crossterms(nSpikes, t1 + threadIdx.x, t2 + ((threadIdx.x+i)%32), m1 + threadIdx.x, m2 + ((threadIdx.x+i)%32), sdf_tau, nSamples, warpSize);
        err += sum_m_N + __shfl_sync(0xffffffff, sum_m_N, threadIdx.x+i);
        err_sdf += (i == HALFWARP && threadIdx.x >= HALFWARP) ? 0 : scalarsqrt(err);
    }

    // Compare delay maps
    for ( int row = 0; row < DMAP_SIZE; row++ ) {
        for ( int col = 0; col < DMAP_SIZE; col++ ) {
            scalar pixel1 = dmap[(row*DMAP_SIZE + col)*NMODELS + offset1];
            scalar pixel2 = ndiag ? dmap[(row*DMAP_SIZE + col)*NMODELS + offset2] : pixel1;
            for ( int i = ndiag ? 0 : 1; i < HALFWARP+1; i++ ) {
                err = pixel1 - __shfl_sync(0xffffffff, pixel2, threadIdx.x+i);
                err_dmaps[i] += (i == HALFWARP && threadIdx.x >= HALFWARP) ? 0 : err*err;
            }
        }
    }
    err_dmaps[0] = ndiag ? sqrt(err_dmaps[0]) : 0;
    for ( int i = 1; i < HALFWARP+1; i++ ) {
        err_dmaps[0] += sqrt(err_dmaps[i]);
    }

    // Collate (using err_trace as collective accumulator)
    err_trace = err_trace * err_weight_trace + err_sdf * err_weight_sdf + err_dmaps[0] * err_weight_dmap;
    err_trace = warpReduceSum(err_trace);
    if ( threadIdx.x == 0 )
        atomicAdd(&errors[blockIdx.z], (scalar)err_trace);
}

/// First pass; one thread per model
/// In single-target mode (GA round, compare vs. data), this immediately compares the voltage trace and the SDF to the target (in filtV, spiketimes_target, spikemarks_target).
/// In multi-target mode (stim selection round), this only produces the filtered voltage trace (written back to dd_timeseries) and spike times/marks in st_out, sm_out.
/// In both modes, a raw delay map is produced in Vx, Vy.
template <bool SINGLETARGET>
__global__ void cl_process_timeseries_kernel(int nSamples, scalar Kfilter, scalar Kfilter2, scalar *filtV, scalar err_weight_trace,
                                             scalar spike_threshold, scalar sdf_tau, scalar sum_m_N, scalar err_weight_sdf, int *st_out, scalar *sm_out,
                                             int delaySize, scalar dmap_low, scalar dmap_step, bool cumulative)
{
    const unsigned int modelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar V, fV = 0, ffV = 0, fn = 0, ffn = 0;
    scalar err = 0;
    unsigned int spike = 0x0;
    int spiketimes[NSPIKES];
    scalar spikemarks[NSPIKES];
    __shared__ uint32_t dmap[DMAP_SIZE * DMAP_SIZE/32][CL_PROCESS_KERNSZ]; // Bitmap. Register relief only, no cross-thread communication.
    scalar Vbuf[DELAYSZ];
    unsigned short dmx = 0;
    int c;
    scalar *pVx =& Vx[modelIdx*DMAP_MAXENTRIES];
    scalar *pVy =& Vy[modelIdx*DMAP_MAXENTRIES];

    for ( int i = 0; i < DMAP_SIZE * DMAP_SIZE/32; i++ )
        dmap[i][threadIdx.x] = 0u;
    // No sync necessary

    for ( int t = 0; t < nSamples; t++ ) {
        // Trace comparison
        V = dd_timeseries[t*NMODELS + modelIdx];
        fV = fV * Kfilter + V;
        fn = fn * Kfilter + 1.0;
        ffV = ffV * Kfilter2 + V;
        ffn = ffn * Kfilter2 + 1.0;
if ( SINGLETARGET ) {
        err += scalarfabs(filtV[t] - fV/fn + ffV/ffn);
} else {
        dd_timeseries[t*NMODELS + modelIdx] = fV/fn - ffV/ffn;
}

        if ( t > delaySize ) {

            // Spike detection
            if ( spike < NSPIKES ) {
                if ( V - Vbuf[t % delaySize] > spike_threshold ) // Spike onset detected
                    spike = spike | SPIKE_DETECTED;
            } else if ( spike & PEAK_DETECTED ) {
                if ( V - Vbuf[t % delaySize] < spike_threshold ) // Spike offset detected
                    spike &= NSPIKES_MASK;
            } else if ( spike & SPIKE_DETECTED ) {
                if ( V < Vbuf[(t-1) % delaySize] ) { // Spike peak detected
                    spike &= NSPIKES_MASK;
                    spikemarks[spike] = spike==0 ? 0 : (spikemarks[spike-1] + 1) * scalarexp((spiketimes[spike-1] - t) / sdf_tau);
                    sum_m_N += spikemarks[spike] + 0.5;
                    spiketimes[spike] = t;
                    spike = (spike+1) | PEAK_DETECTED;
                }
            }

            // Build raw delay map
            c = clip(int((Vbuf[t % delaySize]-dmap_low) / dmap_step), 0, DMAP_SIZE-1) * DMAP_SIZE + clip(int((V-dmap_low) / dmap_step), 0, DMAP_SIZE-1);
            if ( !(dmap[c/32][threadIdx.x] & (1u << (c%32))) ) {
                dmap[c/32][threadIdx.x] |= (1u << (c%32));
                pVx[dmx] = V;
                pVy[dmx] = Vbuf[t % delaySize];
                dmx = (dmx + DMAP_ENTRY_STRIDE) % DMAP_MAXENTRIES;
            }
        }
        Vbuf[t % delaySize] = V;
    }

if ( SINGLETARGET ) {
    scalar err_trace = err_weight_trace * err;
    scalar err_sdf = err_weight_sdf * scalarsqrt(cl_get_sdf_crossterms(spike & NSPIKES_MASK, spiketimes, spiketimes_target, spikemarks, spikemarks_target, sdf_tau, nSamples) + sum_m_N);
    err = err_trace + err_sdf;
    if ( cumulative )
        dd_summary[modelIdx] += err;
    else
        dd_summary[modelIdx] = err;
} else {
    for ( int i = spike & NSPIKES_MASK; i < NSPIKES; i++ )
        spiketimes[i] = nSamples+1;
    __syncwarp();
    for ( int i = 0; i < NSPIKES; i++ ) {
        st_out[i*NMODELS + modelIdx] = spiketimes[i];
        sm_out[i*NMODELS + modelIdx] = spikemarks[i];
    }
}
}

/// Helper for second pass
__device__ void cl_get_smooth_dmap(scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared, scalar *dmap, int stride = 1)
{
    // blockIdx : modelidx
    // threadIdx.z == warpid : Vx/Vy list idx, striding
    // threadIdx x/y : kernel pixel coordinates
    // Note, atomic clashes can't happen within a warp, and are made less likely between warps (I hope) with the DMAP_ENTRY_STRIDE trickery above.
    for ( int i = threadIdx.z; i < DMAP_MAXENTRIES; i += DMAP_STRIDE ) {
        scalar lVx = Vx[blockIdx.x * DMAP_MAXENTRIES + i];
        scalar lVy = Vy[blockIdx.x * DMAP_MAXENTRIES + i];
        if ( lVx >= dmap_low ) {
            const int ix0 = scalarfloor((lVx-dmap_low)/dmap_step) - DMAP_KW/2 + threadIdx.x;
            const int iy0 = scalarfloor((lVy-dmap_low)/dmap_step) - DMAP_KW/2;
#pragma unroll
            for ( int row = threadIdx.y; row < DMAP_KW; row += 32/DMAP_KW ) {
                const scalar dx = ix0 * dmap_step + dmap_low - lVx;
                const scalar dy = (iy0+row) * dmap_step + dmap_low - lVy;
                if ( ix0 >= 0 && ix0 < DMAP_SIZE && iy0+row >= 0 && iy0+row < DMAP_SIZE )
                    atomicAdd(dmap + stride * ((iy0 + row) * DMAP_SIZE + ix0), exp(-(dx*dx + dy*dy)/dmap_2_sigma_squared));
            }
        }
    }
}

/// Second pass (single-target): smoothen delay map and compare to target
__global__ void cl_process_dmaps_kernel_single(scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared, scalar err_weight)
{
    __shared__ scalar dmap[DMAP_SIZE * DMAP_SIZE];
    for ( int i = threadIdx.x + threadIdx.y*DMAP_KW + threadIdx.z*32; i < DMAP_SIZE*DMAP_SIZE; i += 32*DMAP_STRIDE )
        dmap[i] = scalar(0);
    __syncthreads();

    // build smooth delay map in shared memory
    cl_get_smooth_dmap(dmap_low, dmap_step, dmap_2_sigma_squared, dmap);

    // Compare to target
    scalar err = 0;
    for ( int i = threadIdx.x + threadIdx.y*DMAP_KW + threadIdx.z*32; i < DMAP_SIZE*DMAP_SIZE; i += 32*DMAP_STRIDE ) {
        scalar lerr = dmap_target[i] - dmap[i];
        err += lerr*lerr;
    }
    __syncthreads();
    err = warpReduceSum(err);
    if ( threadIdx.x == 0 && threadIdx.y == 0 )
        dmap[threadIdx.z] = err;
    __syncthreads();
    if ( threadIdx.z == 0 ) {
        err = dmap[threadIdx.x + threadIdx.y*DMAP_KW];
        __syncwarp();
        err = warpReduceSum(err, DMAP_STRIDE);
        if ( threadIdx.x == 0 && threadIdx.y == 0 ) {
            dd_summary[blockIdx.x] += scalarsqrt(err) * err_weight;
        }
    }
}

/// Second pass (multi-target): smoothen delay map for later use
__global__ void cl_process_dmaps_kernel_multi(scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared, scalar *dmaps)
{
    cl_get_smooth_dmap(dmap_low, dmap_step, dmap_2_sigma_squared, dmaps + blockIdx.x, NMODELS);
}

/// CPU-based target timeseries processing. Produces filtered voltage trace, spike times and marks, and a smooth dmap.
scalar cl_process_timeseries_target(int nSamples, scalar Kfilter, scalar Kfilter2,
                                    scalar spike_threshold, scalar sdf_tau,
                                    int delaySize, scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared,
                                    scalar *filtV, int *spiketimes, scalar *spikemarks, scalar *dmap_smooth, scalar *target)
{
    scalar V, fV = 0, ffV = 0, fn = 0, ffn = 0;
    unsigned int spike = 0x0;
    uint32_t dmap[DMAP_SIZE * DMAP_SIZE/32];
    scalar Vbuf[DELAYSZ];
    unsigned short dmx = 0;
    int c;
    scalar Vx_target[DMAP_MAXENTRIES], Vy_target[DMAP_MAXENTRIES];
    scalar sdf_target_sum_m_norm = 0;

    for ( int i = 0; i < DMAP_SIZE * DMAP_SIZE/32; i++ )
        dmap[i] = 0u;

    for ( int t = 0; t < nSamples; t++ ) {
        // Trace comparison
        V = target[t];
        fV = fV * Kfilter + V;
        fn = fn * Kfilter + 1.0;
        ffV = ffV * Kfilter2 + V;
        ffn = ffn * Kfilter2 + 1.0;
        filtV[t] = fV/fn - ffV/ffn;

        if ( t > delaySize ) {

            // Spike detection
            if ( spike < NSPIKES ) {
                if ( V - Vbuf[t % delaySize] > spike_threshold ) // Spike onset detected
                    spike = spike | SPIKE_DETECTED;
            } else if ( spike & PEAK_DETECTED ) {
                if ( V - Vbuf[t % delaySize] < spike_threshold ) // Spike offset detected
                    spike &= NSPIKES_MASK;
            } else if ( spike & SPIKE_DETECTED ) {
                if ( V < Vbuf[(t-1) % delaySize] ) { // Spike peak detected
                    spike &= NSPIKES_MASK;
                    spikemarks[spike] = spike==0 ? 0 : (spikemarks[spike-1] + 1) * scalarexp((spiketimes[spike-1] - t) / sdf_tau);
                    spiketimes[spike] = t;
                    sdf_target_sum_m_norm += spikemarks[spike] + 0.5;
                    spike = (spike+1) | PEAK_DETECTED;
                }
            }

            // Build raw delay map
            c = clip(int((Vbuf[t % delaySize]-dmap_low) / dmap_step), 0, DMAP_SIZE-1) * DMAP_SIZE + clip(int((V-dmap_low) / dmap_step), 0, DMAP_SIZE-1);
            if ( !(dmap[c/32] & (1u << (c%32))) ) {
                dmap[c/32] |= (1u << (c%32));
                Vx_target[dmx] = V;
                Vy_target[dmx] = Vbuf[t % delaySize];
                ++dmx;
            }
        }
        Vbuf[t % delaySize] = V;
    }

    // Pad spiketimes with past-the-end values
    for ( int i = spike & NSPIKES_MASK; i < NSPIKES; i++ )
        spiketimes[i] = nSamples+1;

    // Pre-fill smooth delay map with zeroes
    for ( int iy = 0; iy < DMAP_SIZE; ++iy )
        for ( int ix = 0; ix < DMAP_SIZE; ++ix )
            dmap_smooth[iy * DMAP_SIZE + ix] = 0;

    // Produce smooth delay map
    for ( int i = 0; i < dmx; ++i ) {
        scalar lVx = Vx_target[i];
        scalar lVy = Vy_target[i];
        const int ix0 = scalarfloor((lVx-dmap_low)/dmap_step) - DMAP_KW/2;
        const int iy0 = scalarfloor((lVy-dmap_low)/dmap_step) - DMAP_KW/2;
        for ( int iy = max(0,iy0); iy < min(DMAP_SIZE-1, iy0 + DMAP_KW); ++iy ) {
            for ( int ix = max(0,ix0); ix < min(DMAP_SIZE-1, ix0 + DMAP_KW); ++ix ) {
                const scalar dx = ix * dmap_step + dmap_low - lVx;
                const scalar dy = iy * dmap_step + dmap_low - lVy;
                dmap_smooth[iy * DMAP_SIZE + ix] += exp(-(dx*dx + dy*dy)/dmap_2_sigma_squared);
            }
        }
    }

    return sdf_target_sum_m_norm;
}

extern "C" void cl_compare_to_target(int nSamples, ClosedLoopData d, double dt, bool reset_summary, scalar *target)
{
    int delaySize = min(DELAYSZ, int(round(d.tDelay/dt)));
    unsigned int filtV_size_h = filtV_size;
    resizeHostArray(h_filtV, filtV_size_h, nSamples);
    scalar sdf_target_sum_m_norm = cl_process_timeseries_target(nSamples, d.Kfilter, d.Kfilter2, d.spike_threshold, d.sdf_tau/dt, delaySize, d.dmap_low, d.dmap_step, 2*d.dmap_sigma*d.dmap_sigma,
                                                                h_filtV, h_spiketimes_target, h_spikemarks_target, h_dmap_target, target);

    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(spiketimes_target, h_spiketimes_target, NSPIKES * sizeof(int)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(spikemarks_target, h_spikemarks_target, NSPIKES * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dmap_target, h_dmap_target, DMAP_SIZE*DMAP_SIZE * sizeof(scalar)));

    resizeArray(d_filtV, filtV_size, nSamples);
    CHECK_CUDA_ERRORS(cudaMemcpy(d_filtV, h_filtV, nSamples * sizeof(scalar), cudaMemcpyHostToDevice));

    if ( Vx_size != DMAP_MAXENTRIES * NMODELS || Vy_size != DMAP_MAXENTRIES * NMODELS ) {
        resizeArray(d_Vx, Vx_size, DMAP_MAXENTRIES * NMODELS);
        resizeArray(d_Vy, Vy_size, DMAP_MAXENTRIES * NMODELS);
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(Vx, &d_Vx, sizeof(scalar*)));
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(Vy, &d_Vy, sizeof(scalar*)));
    }
    thrust::device_ptr<scalar> t_Vx = thrust::device_pointer_cast(d_Vx);
    thrust::fill(t_Vx, t_Vx + DMAP_MAXENTRIES * NMODELS, d.dmap_low - 1.);

    cl_process_timeseries_kernel<true><<<NMODELS/CL_PROCESS_KERNSZ, CL_PROCESS_KERNSZ>>>(nSamples, d.Kfilter, d.Kfilter2, d_filtV, d.err_weight_trace/nSamples,
                                                                                         d.spike_threshold*dt*delaySize, d.sdf_tau/dt, sdf_target_sum_m_norm, d.err_weight_sdf, nullptr, nullptr,
                                                                                         delaySize, d.dmap_low, d.dmap_step, !reset_summary);

    cl_process_dmaps_kernel_single<<<NMODELS, dim3(DMAP_KW, 32/DMAP_KW, DMAP_STRIDE)>>>(d.dmap_low, d.dmap_step, 2*d.dmap_sigma*d.dmap_sigma, d.err_weight_dmap/DMAP_SIZE);
}

extern "C" std::vector<scalar> cl_compare_models(int nStims, unsigned int nSamples, ClosedLoopData d, double dt)
{
    int delaySize = min(DELAYSZ, int(round(d.tDelay/dt)));
    resizeArray(spiketimes_models, spiketimes_size, NSPIKES * NMODELS);
    resizeArray(spikemarks_models, spikemarks_size, NSPIKES * NMODELS);
    resizeArray(d_dmaps, dmaps_size, NMODELS * DMAP_SIZE * DMAP_SIZE);

    if ( Vx_size != DMAP_MAXENTRIES * NMODELS || Vy_size != DMAP_MAXENTRIES * NMODELS ) {
        resizeArray(d_Vx, Vx_size, DMAP_MAXENTRIES * NMODELS);
        resizeArray(d_Vy, Vy_size, DMAP_MAXENTRIES * NMODELS);
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(Vx, &d_Vx, sizeof(scalar*)));
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(Vy, &d_Vy, sizeof(scalar*)));
    }
    thrust::device_ptr<scalar> tmp = thrust::device_pointer_cast(d_Vx);
    thrust::fill(tmp, tmp + DMAP_MAXENTRIES * NMODELS, d.dmap_low - 1.);

    cl_process_timeseries_kernel<false><<<NMODELS/CL_PROCESS_KERNSZ, CL_PROCESS_KERNSZ>>>(nSamples, d.Kfilter, d.Kfilter2, nullptr, d.err_weight_trace/nSamples,
                                                                                          d.spike_threshold*dt*delaySize, d.sdf_tau/dt, 0, d.err_weight_sdf, spiketimes_models, spikemarks_models,
                                                                                          delaySize, d.dmap_low, d.dmap_step, false);

    cl_process_dmaps_kernel_multi<<<NMODELS, dim3(DMAP_KW, 32/DMAP_KW, DMAP_STRIDE)>>>(d.dmap_low, d.dmap_step, 2*d.dmap_sigma*d.dmap_sigma, d_dmaps);

    int nTraces = NMODELS / nStims;
    int nUnits = nTraces / 32;
    int nComparisonsPerStim = nTraces*(nTraces-1)/2;
    std::vector<scalar> means(nStims);

    tmp = thrust::device_pointer_cast(d_prof_error);
    thrust::fill(tmp, tmp + nStims, scalar(0));

    dim3 grid(nUnits, nUnits, nStims);
    compare_models_kernel<<<grid, 32>>>(nTraces, nSamples, spiketimes_models, spikemarks_models, d_dmaps, d.sdf_tau/dt, d.err_weight_trace/nSamples, d.err_weight_sdf, d.err_weight_dmap/DMAP_SIZE, d_prof_error);

    CHECK_CUDA_ERRORS(cudaMemcpy(means.data(), d_prof_error, nStims*sizeof(scalar), cudaMemcpyDeviceToHost));

    for ( scalar &m : means )
        m /= nComparisonsPerStim;
    return means;
}

extern "C" scalar cl_dmap_hi(scalar dmap_low, scalar dmap_step) { return dmap_low + DMAP_SIZE*dmap_step; }
