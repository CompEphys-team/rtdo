#include "lib_definitions.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/fill.h>


#define CL_PROCESS_KERNSZ 32
#define NSPIKES 32
#define NSPIKES_MASK 0xff
#define SPIKE_DETECTED 0x100
#define PEAK_DETECTED 0x200
#define SDF_KERNEL_MAXWIDTH 1024
#define DELAYSZ 16
#define DMAP_MAXENTRIES 1024
#define DMAP_SIZE 64
#define DMAP_STRIDE 16
#define DMAP_KW 16 /* kernel width */

scalar *h_filtV = nullptr, *d_filtV = nullptr;
unsigned int filtV_size = 0;

int *spiketimes_models = nullptr;
unsigned int spiketimes_size = 0;

scalar *d_dmaps = nullptr;
unsigned int dmaps_size = 0;

scalar *d_Vx = nullptr, *d_Vy = nullptr;
__constant__ scalar *Vx = nullptr, *Vy = nullptr;
unsigned int Vx_size = 0, Vy_size = 0;

scalar *d_partial_errors = nullptr, *h_partial_errors = nullptr;
unsigned int partial_errors_size = 0, partial_errors_hsize = 0;

__constant__ int spiketimes_target[NSPIKES];
__constant__ scalar dmap_target[DMAP_SIZE * DMAP_SIZE];
int h_spiketimes_target[NSPIKES];
scalar h_dmap_target[DMAP_SIZE * DMAP_SIZE];

__constant__ scalar sdf_kernel[SDF_KERNEL_MAXWIDTH];
double latest_sdf_kernel_sigma = 0;


__device__ inline scalar get_sdf(const int iT, const int * const tSpk, short &start, short &stop, const int tMax, const int kernel_width)
{
    scalar ret = 0;
    while ( iT - tSpk[start] > kernel_width && start < NSPIKES )
        ++start;
    while ( tSpk[stop] - iT < kernel_width && tSpk[stop] < tMax && stop < NSPIKES )
        ++stop;
    for ( short idx = start; idx < stop; idx++ ) {
        int deltaT = __sad(iT, tSpk[idx], 0); // == abs(t - t1[idx])
        ret += sdf_kernel[deltaT];
    }
    return ret;
}

/// For launch details see compare_models_kernel()
__global__ void compare_model_sdf_kernel(int nTraces, int nSamples, int kernel_width, int *st_in, scalar err_weight_sdf, scalar *errors)
{
    if ( blockIdx.y > blockIdx.x )
        return; // Do not calculate the duplicate below-diagonal units

    const int offset1 = blockIdx.z*nTraces + blockIdx.x*warpSize + threadIdx.x;
    const int offset2 = blockIdx.z*nTraces + blockIdx.y*warpSize + threadIdx.x;
    const bool diagonal = (offset1 == offset2);

    int t1[NSPIKES], t2[NSPIKES];
    scalar sdf1, sdf2;
    short start1 = 0, stop1 = 0;
    short start2 = 0, stop2 = 0;
    double err_sdf = 0;
    scalar err;

    // load spike times
    for ( int i = 0; i < NSPIKES; i++ ) {
        t1[i] = st_in[i*NMODELS + offset1];
        t2[i] = diagonal ? t1[i] : st_in[i*NMODELS + offset2];
        if ( t1[i] > nSamples && t2[i] > nSamples )
            break;
    }
    __syncwarp();

    // Compare
    for ( int t = 0; t < nSamples; t++ ) {
        sdf1 = get_sdf(t, t1, start1, stop1, nSamples, kernel_width);
        sdf2 = get_sdf(t, t2, start2, stop2, nSamples, kernel_width);
        for ( int i = diagonal ? 1 : 0; i < 17; i++ ) {
            err = scalarfabs(sdf1 - __shfl_sync(0xffffffff, sdf2, threadIdx.x+i));
            err_sdf += (i == 16 && threadIdx.x >= 16) ? 0 : err;
        }
    }

    // Collate within block
    err_sdf = warpReduceSum(err_sdf * err_weight_sdf);
    if ( threadIdx.x == 0 ) {
        atomicAdd(&errors[blockIdx.z], scalar(err_sdf));
        atomicAdd(&errors[2*gridDim.z + blockIdx.z], scalar(err_sdf));
    }
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
__global__ void compare_models_kernel(int nTraces, int nSamples, scalar *dmap, scalar err_weight_trace, scalar err_weight_dmap, scalar *errors)
{
    if ( blockIdx.y > blockIdx.x )
        return; // Do not calculate the duplicate below-diagonal units

    constexpr int HALFWARP = 16;
    const int offset1 = blockIdx.z*nTraces + blockIdx.x*warpSize + threadIdx.x;
    const int offset2 = blockIdx.z*nTraces + blockIdx.y*warpSize + threadIdx.x;
    const bool ndiag = (offset1 != offset2);

    scalar err;
    double err_trace = 0;
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

    // Collate within block
    err_trace = warpReduceSum(err_trace * err_weight_trace);
    err_dmaps[0] = warpReduceSum(err_dmaps[0] * err_weight_dmap);

    if ( threadIdx.x == 0 ) {
        atomicAdd(&errors[blockIdx.z], scalar(err_trace + err_dmaps[0]));
        atomicAdd(&errors[gridDim.z + blockIdx.z], scalar(err_trace));
        atomicAdd(&errors[3*gridDim.z + blockIdx.z], scalar(err_dmaps[0]));
    }
}

/// First pass; one thread per model
/// In single-target mode (GA round, compare vs. data), this immediately compares the voltage trace and the SDF to the target (in filtV, spiketimes_target).
/// In multi-target mode (stim selection round), this only produces the filtered voltage trace (written back to dd_timeseries) and spike times in st_out.
/// In both modes, a raw delay map is produced in Vx, Vy.
template <bool SINGLETARGET>
__global__ void cl_process_timeseries_kernel(int nSamples, scalar Kfilter, scalar Kfilter2, scalar *filtV, scalar err_weight_trace,
                                             scalar spike_threshold, int sdf_kernel_width, scalar err_weight_sdf, int *st_out,
                                             int delaySize, scalar dmap_low, scalar dmap_step, bool cumulative, scalar *err_partial)
{
    const unsigned int modelIdx = blockIdx.x * blockDim.x + threadIdx.x;
    scalar V, fV = 0, ffV = 0, fn = 0, ffn = 0;
    double err_trace = 0;
    unsigned int spike = 0x0;
    int spiketimes[NSPIKES];
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
        V = dd_timeseries[t*NMODELS + modelIdx];

        // Build raw delay map
        if ( t > delaySize ) {
            c = clip(int((Vbuf[t % delaySize]-dmap_low) / dmap_step), 0, DMAP_SIZE-1) * DMAP_SIZE + clip(int((V-dmap_low) / dmap_step), 0, DMAP_SIZE-1);
            if ( !(dmap[c/32][threadIdx.x] & (1u << (c%32))) ) {
                dmap[c/32][threadIdx.x] |= (1u << (c%32));
                pVx[dmx] = V;
                pVy[dmx] = Vbuf[t % delaySize];
                ++dmx;
            }
        }
        Vbuf[t % delaySize] = V;

        // Filter trace
        fV = fV * Kfilter + V;
        fn = fn * Kfilter + 1.0;
        ffV = ffV * Kfilter2 + V;
        ffn = ffn * Kfilter2 + 1.0;
        V = fV/fn - ffV/ffn;
if ( SINGLETARGET ) {
        err_trace += abs(filtV[t] - V);
} else {
        dd_timeseries[t*NMODELS + modelIdx] = V;
}

        // Spike detection
        if ( t > delaySize ) {
            if ( spike < NSPIKES ) {
                if ( V > spike_threshold ) { // Spike onset detected
                    spiketimes[spike] = t;
                    spike = (spike+1) | SPIKE_DETECTED;
                }
            } else if ( spike & SPIKE_DETECTED ) {
                if ( V < spike_threshold ) { // Spike offset detected
                    spike &= NSPIKES_MASK;
                }
            }
        }
    }

    for ( spike &= NSPIKES_MASK; spike < NSPIKES; spike++ )
        spiketimes[spike] = nSamples + 1;
    __syncwarp();

if ( SINGLETARGET ) {

    // Compare SDF
    short start1 = 0, stop1 = 0;
    short start2 = 0, stop2 = 0;
    double err_sdf = 0;
    for ( int t = 0; t < nSamples; t++ ) {
        scalar sdf1 = get_sdf(t, spiketimes, start1, stop1, nSamples, sdf_kernel_width);
        scalar sdf2 = get_sdf(t, spiketimes_target, start2, stop2, nSamples, sdf_kernel_width);
        err_sdf += scalarfabs(sdf1 - sdf2);
    }
    err_sdf *= err_weight_sdf;
    err_partial[NMODELS + modelIdx] = err_sdf;

    err_trace *= err_weight_trace;
    err_partial[modelIdx] = err_trace;

    if ( cumulative )
        dd_summary[modelIdx] += err_trace + err_sdf;
    else
        dd_summary[modelIdx] = err_trace + err_sdf;
} else {
    for ( int i = 0; i < NSPIKES; i++ ) {
        st_out[i*NMODELS + modelIdx] = spiketimes[i];
    }
}
}

/// Helper for second pass
__device__ void cl_get_smooth_dmap(scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared, scalar *dmap, int stride = 1)
{
    // blockIdx : modelidx
    // threadIdx.z == warpid : Vx/Vy list idx, striding
    // threadIdx x/y : kernel pixel coordinates
    // Note, atomic clashes can't happen within a warp.
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
__global__ void cl_process_dmaps_kernel_single(scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared, scalar err_weight, scalar *err_partial)
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
            err = scalarsqrt(err) * err_weight;
            err_partial[2*NMODELS + blockIdx.x] = err;
            dd_summary[blockIdx.x] += err;
        }
    }
}

/// Second pass (multi-target): smoothen delay map for later use
__global__ void cl_process_dmaps_kernel_multi(scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared, scalar *dmaps)
{
    cl_get_smooth_dmap(dmap_low, dmap_step, dmap_2_sigma_squared, dmaps + blockIdx.x, NMODELS);
}

/// CPU-based target timeseries processing. Produces filtered voltage trace, spike times, and a smooth dmap.
void cl_process_timeseries_target(int nSamples, scalar Kfilter, scalar Kfilter2,
                                    scalar spike_threshold,
                                    int delaySize, scalar dmap_low, scalar dmap_step, scalar dmap_2_sigma_squared,
                                    scalar *filtV, int *spiketimes, scalar *dmap_smooth, scalar *target)
{
    scalar V, fV = 0, ffV = 0, fn = 0, ffn = 0;
    unsigned int spike = 0x0;
    uint32_t dmap[DMAP_SIZE * DMAP_SIZE/32];
    scalar Vbuf[DELAYSZ];
    unsigned short dmx = 0;
    int c;
    scalar Vx_target[DMAP_MAXENTRIES], Vy_target[DMAP_MAXENTRIES];

    for ( int i = 0; i < DMAP_SIZE * DMAP_SIZE/32; i++ )
        dmap[i] = 0u;

    for ( int t = 0; t < nSamples; t++ ) {
        V = target[t];

        // Build raw delay map
        if ( t > delaySize ) {
            c = clip(int((Vbuf[t % delaySize]-dmap_low) / dmap_step), 0, DMAP_SIZE-1) * DMAP_SIZE + clip(int((V-dmap_low) / dmap_step), 0, DMAP_SIZE-1);
            if ( !(dmap[c/32] & (1u << (c%32))) ) {
                dmap[c/32] |= (1u << (c%32));
                Vx_target[dmx] = V;
                Vy_target[dmx] = Vbuf[t % delaySize];
                ++dmx;
            }
        }
        Vbuf[t % delaySize] = V;

        // Filter trace
        fV = fV * Kfilter + V;
        fn = fn * Kfilter + 1.0;
        ffV = ffV * Kfilter2 + V;
        ffn = ffn * Kfilter2 + 1.0;
        V = fV/fn - ffV/ffn;
        filtV[t] = V;

        // Spike detection
        if ( t > delaySize ) {
            if ( spike < NSPIKES ) {
                if ( V > spike_threshold ) { // Spike onset detected
                    spiketimes[spike] = t;
                    spike = (spike+1) | SPIKE_DETECTED;
                }
            } else if ( spike & SPIKE_DETECTED ) {
                if ( V < spike_threshold ) { // Spike offset detected
                    spike &= NSPIKES_MASK;
                }
            }
        }
    }

    // Pad spiketimes with past-the-end values
    for ( spike &= NSPIKES_MASK; spike < NSPIKES; spike++ )
        spiketimes[spike] = nSamples + 1;

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
}

int generate_sdf_kernel(double sigma)
{
    int kernel_width = std::min(int(round(4 * sigma)), SDF_KERNEL_MAXWIDTH);
    if ( sigma != latest_sdf_kernel_sigma ) {
        std::vector<scalar> kernel(kernel_width);
        double two_variance = 2.0*sigma*sigma;
        double offset = exp(-(kernel_width+1)*(kernel_width+1) / two_variance);
        for ( int i = 0; i < kernel_width; i++ ) {
            kernel[i] = exp(-i*i / two_variance) - offset;
        }
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(sdf_kernel, kernel.data(), kernel_width * sizeof(scalar)));
        latest_sdf_kernel_sigma = sigma;
    }
    return kernel_width;
}

extern "C" scalar *cl_compare_to_target(int nSamples, ClosedLoopData d, double dt, bool reset_summary, scalar *target)
{
    int delaySize = min(DELAYSZ, int(round(d.tDelay/dt)));
    unsigned int filtV_size_h = filtV_size;
    resizeHostArray(h_filtV, filtV_size_h, nSamples);

    cl_process_timeseries_target(nSamples, d.Kfilter, d.Kfilter2, d.spike_threshold, delaySize, d.dmap_low, d.dmap_step, 2*d.dmap_sigma*d.dmap_sigma,
                                 h_filtV, h_spiketimes_target, h_dmap_target, target);

    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(spiketimes_target, h_spiketimes_target, NSPIKES * sizeof(int)));
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

    resizeArray(d_partial_errors, partial_errors_size, 3*NMODELS);

    int sdf_kernel_width = generate_sdf_kernel(d.sdf_tau/dt);

    cl_process_timeseries_kernel<true><<<NMODELS/CL_PROCESS_KERNSZ, CL_PROCESS_KERNSZ>>>(nSamples, d.Kfilter, d.Kfilter2, d_filtV, d.err_weight_trace/nSamples,
                                                                                         d.spike_threshold*dt*delaySize, sdf_kernel_width, d.err_weight_sdf/nSamples, nullptr,
                                                                                         delaySize, d.dmap_low, d.dmap_step, !reset_summary, d_partial_errors);

    cl_process_dmaps_kernel_single<<<NMODELS, dim3(DMAP_KW, 32/DMAP_KW, DMAP_STRIDE)>>>(d.dmap_low, d.dmap_step, 2*d.dmap_sigma*d.dmap_sigma, d.err_weight_dmap/DMAP_SIZE, d_partial_errors);

    resizeHostArray(h_partial_errors, partial_errors_hsize, 3*NMODELS);
    CHECK_CUDA_ERRORS(cudaMemcpy(h_partial_errors, d_partial_errors, 3*NMODELS*sizeof(scalar), cudaMemcpyDeviceToHost));
    return h_partial_errors;
}

extern "C" std::vector<std::tuple<scalar, scalar, scalar, scalar>> cl_compare_models(int nStims, unsigned int nSamples, ClosedLoopData d, double dt)
{
    int delaySize = min(DELAYSZ, int(round(d.tDelay/dt)));
    resizeArray(spiketimes_models, spiketimes_size, NSPIKES * NMODELS);
    resizeArray(d_dmaps, dmaps_size, NMODELS * DMAP_SIZE * DMAP_SIZE);

    if ( Vx_size != DMAP_MAXENTRIES * NMODELS || Vy_size != DMAP_MAXENTRIES * NMODELS ) {
        resizeArray(d_Vx, Vx_size, DMAP_MAXENTRIES * NMODELS);
        resizeArray(d_Vy, Vy_size, DMAP_MAXENTRIES * NMODELS);
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(Vx, &d_Vx, sizeof(scalar*)));
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(Vy, &d_Vy, sizeof(scalar*)));
    }
    thrust::device_ptr<scalar> tmp = thrust::device_pointer_cast(d_Vx);
    thrust::fill(tmp, tmp + DMAP_MAXENTRIES * NMODELS, d.dmap_low - 1.);

    resizeArray(d_partial_errors, partial_errors_size, 4*nStims);
    resizeHostArray(h_partial_errors, partial_errors_hsize, 4*nStims);

    cl_process_timeseries_kernel<false><<<NMODELS/CL_PROCESS_KERNSZ, CL_PROCESS_KERNSZ>>>(nSamples, d.Kfilter, d.Kfilter2, nullptr, d.err_weight_trace/nSamples,
                                                                                          d.spike_threshold*dt*delaySize, 0, d.err_weight_sdf/nSamples, spiketimes_models,
                                                                                          delaySize, d.dmap_low, d.dmap_step, false, nullptr);

    cl_process_dmaps_kernel_multi<<<NMODELS, dim3(DMAP_KW, 32/DMAP_KW, DMAP_STRIDE)>>>(d.dmap_low, d.dmap_step, 2*d.dmap_sigma*d.dmap_sigma, d_dmaps);

    int sdf_kernel_width = generate_sdf_kernel(d.sdf_tau/dt);

    int nTraces = NMODELS / nStims;
    int nUnits = nTraces / 32;
    int nComparisonsPerStim = nTraces*(nTraces-1)/2;
    std::vector<std::tuple<scalar, scalar, scalar, scalar>> means(nStims);

    tmp = thrust::device_pointer_cast(d_partial_errors);
    thrust::fill(tmp, tmp + 4*nStims, scalar(0));

    dim3 grid(nUnits, nUnits, nStims);
    compare_models_kernel<<<grid, 32>>>(nTraces, nSamples, d_dmaps, d.err_weight_trace/nSamples, d.err_weight_dmap/DMAP_SIZE, d_partial_errors);
    compare_model_sdf_kernel<<<grid, 32>>>(nTraces, nSamples, sdf_kernel_width, spiketimes_models, d.err_weight_sdf/nSamples, d_partial_errors);

    CHECK_CUDA_ERRORS(cudaMemcpy(h_partial_errors, d_partial_errors, 4*nStims*sizeof(scalar), cudaMemcpyDeviceToHost));
    for ( int i = 0; i < nStims; i++ ) {
        scalar mean = h_partial_errors[i] / nComparisonsPerStim;
        scalar trace = h_partial_errors[nStims + i] / nComparisonsPerStim;
        scalar sdf = h_partial_errors[2*nStims + i] / nComparisonsPerStim;
        scalar dmap = h_partial_errors[3*nStims + i] / nComparisonsPerStim;
        means[i] = std::make_tuple(mean, trace, sdf, dmap);
    }

    return means;
}

extern "C" scalar cl_dmap_hi(scalar dmap_low, scalar dmap_step) { return dmap_low + DMAP_SIZE*dmap_step; }
