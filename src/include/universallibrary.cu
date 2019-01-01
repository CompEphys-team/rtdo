#ifndef UNIVERSAL_CU
#define UNIVERSAL_CU

#include "universallibrary.h"
#include "cuda_helper.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

static scalar *target = nullptr, *d_target = nullptr;
static __constant__ scalar *dd_target = nullptr;
static unsigned int target_size = 0, latest_target_size = 0;

static scalar *timeseries = nullptr, *d_timeseries = nullptr;
static __constant__ scalar *dd_timeseries = nullptr;
static unsigned int timeseries_size = 0, latest_timeseries_size = 0;

static __constant__ iStimulation singular_stim;
static __constant__ iObservations singular_obs;

static __constant__ scalar singular_clampGain;
static __constant__ scalar singular_accessResistance;
static __constant__ int singular_iSettleDuration;
static __constant__ scalar singular_Imax;
static __constant__ scalar singular_dt;

static __constant__ size_t singular_targetOffset;

// profiler memory space
static constexpr unsigned int NPAIRS = NMODELS/2;
static scalar *d_gradient;
static constexpr unsigned int gradientSz = NPAIRS * (NPAIRS - 1); // No diagonal

// elementary effects wg / clustering memory space
static scalar *clusters = nullptr, *d_clusters = nullptr;
static unsigned int clusters_size = 0;

static int *clusterLen = nullptr, *d_clusterLen = nullptr;
static unsigned int clusterLen_size = 0;

static unsigned int *d_clusterMasks = nullptr;
static unsigned int clusterMasks_size = 0;

static scalar *clusterCurrent = nullptr, *d_clusterCurrent = nullptr;
static unsigned int clusterCurrent_size = 0;

static scalar *sections = nullptr, *d_sections = nullptr;
static unsigned int sections_size = 0;

static scalar *d_currents= nullptr;
static unsigned int currents_size = 0;

static iObservations *clusterObs = nullptr, *d_clusterObs = nullptr;
static unsigned int clusterObs_size = 0;

static Bubble *bubbles = nullptr, *d_bubbles = nullptr;
static unsigned int bubbles_size = 0;

static __constant__ scalar deltabar[NPARAMS];

void libInit(UniversalLibrary &lib, UniversalLibrary::Pointers &pointers)
{
    pointers.pushV = [](void *hostptr, void *devptr, size_t size){
        CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostptr, size, cudaMemcpyHostToDevice))
    };
    pointers.pullV = [](void *hostptr, void *devptr, size_t size){
        CHECK_CUDA_ERRORS(cudaMemcpy(hostptr, devptr, size, cudaMemcpyDeviceToHost));
    };

    pointers.target =& target;
    pointers.output =& timeseries;

    pointers.clusters =& clusters;
    pointers.clusterCurrent =& clusterCurrent;
    pointers.clusterPrimitives =& sections;
    pointers.clusterObs =& clusterObs;

    pointers.bubbles =& bubbles;

    allocateMem();
    initialize();

    cudaGetSymbolAddress((void **)&lib.stim.singular_v, singular_stim);
    cudaGetSymbolAddress((void **)&lib.obs.singular_v, singular_obs);

    cudaGetSymbolAddress((void **)&lib.clampGain.singular_v, singular_clampGain);
    cudaGetSymbolAddress((void **)&lib.accessResistance.singular_v, singular_accessResistance);
    cudaGetSymbolAddress((void **)&lib.iSettleDuration.singular_v, singular_iSettleDuration);
    cudaGetSymbolAddress((void **)&lib.Imax.singular_v, singular_Imax);
    cudaGetSymbolAddress((void **)&lib.dt.singular_v, singular_dt);

    cudaGetSymbolAddress((void **)&lib.targetOffset.singular_v, singular_targetOffset);

    CHECK_CUDA_ERRORS(cudaMalloc(&d_gradient, gradientSz * sizeof(scalar)));
}

extern "C" void libExit(UniversalLibrary::Pointers &pointers)
{
    freeMem();
    pointers.pushV = pointers.pullV = nullptr;
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

extern "C" void resizeTarget(size_t newSize)
{
    resizeArrayPair(target, d_target, target_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_target, &d_target, sizeof(scalar*)));
    latest_target_size = newSize;
}

extern "C" void pushTarget()
{
    CHECK_CUDA_ERRORS(cudaMemcpy(d_target, target, latest_target_size * sizeof(scalar), cudaMemcpyHostToDevice))
}

extern "C" void resizeOutput(size_t newSize)
{
    resizeArrayPair(timeseries, d_timeseries, timeseries_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_timeseries, &d_timeseries, sizeof(scalar*)));
    latest_timeseries_size = newSize;
}

extern "C" void pullOutput()
{
    CHECK_CUDA_ERRORS(cudaMemcpy(timeseries, d_timeseries, latest_timeseries_size * sizeof(scalar), cudaMemcpyDeviceToHost))
}


template <typename T>
__device__ inline T warpReduceSum(T val, int cutoff = warpSize)
{
    for ( int offset = 1; offset < cutoff; offset *= 2 )
        val += __shfl_xor_sync(0xffffffff, val, offset);
    return val;
}

template <>
__device__ inline Parameters warpReduceSum<Parameters>(Parameters val, int cutoff)
{
    Parameters addend;
    for ( int offset = 1; offset < cutoff; offset *= 2 ) {
        addend.shfl(val, (threadIdx.x&31) ^ offset, cutoff);
        val += addend;
    }
    return val;
}



/// ******************************************************************************************************************************
///  >============================     Profiler kernel & host function      ====================================================<
/// ******************************************************************************************************************************

// Compute the current deviation of both tuned and untuned models against each tuned model
// Models are interleaved (even id = tuned, odd id = detuned) in SamplingProfiler
__global__ void compute_gradient(int nSamples, int stride, scalar *targetParam, scalar *gradient)
{
    unsigned int xThread = blockIdx.x * blockDim.x + threadIdx.x; // probe
    unsigned int yThread = blockIdx.y * blockDim.y + threadIdx.y; // reference
    unsigned int x,y;
    if ( xThread < yThread ) { // transpose subdiagonal half of the top-left quadrant to run on the supradiagonal half of bottom-right quadrant
        // the coordinate transformation is equivalent to squashing the bottom-right supradiagonal triangle to the left border,
        // then flipping it up across the midline.
        x = xThread + NPAIRS - yThread; // xnew = x + n-y
        y = NPAIRS - yThread - 1;       // ynew = n-y - 1
    } else {
        x = xThread;
        y = yThread;
    }

    scalar err_tx_ty = 0., err_tx_dy = 0., err_dx_ty = 0., err;
    int i = 0;
    for ( ; i < nSamples; i += stride ) {
        scalar xval = dd_timeseries[2*x + NMODELS*i];
        scalar yval = dd_timeseries[2*y + NMODELS*i];

        err = xval - yval;
        err_tx_ty += err*err;

        err = xval - dd_timeseries[2*y+1 + NMODELS*i];
        err_tx_dy += err*err;

        err = yval - dd_timeseries[2*x+1 + NMODELS*i];
        err_dx_ty += err*err;
    }

    i = nSamples/stride; // Using i as nSamplesUsed
    err_tx_ty = std::sqrt(err_tx_ty / i);
    err_tx_dy = std::sqrt(err_tx_dy / i);
    err_dx_ty = std::sqrt(err_dx_ty / i);

    if ( x != y ) { // Ignore diagonal (don't probe against self)
        // invert sign as appropriate, such that detuning in the direction of the reference is reported as positive
        i = (1 - 2 * (targetParam[2*x] < targetParam[2*y])); // using i as sign

        // fractional change in error ( (d_err-t_err)/t_err) "how much does the error improve by detuning, relative to total error?")
        err = ((err_dx_ty / err_tx_ty) - 1) * i;

        // Put invalid values to the end of the scale, positive or negative; heuristically balance both sides
        if ( ::isnan(err) )
            err = i * SCALAR_MAX;

        // Addressing: Squish the diagonal out to prevent extra zeroes
        gradient[xThread + NPAIRS*yThread - yThread - (xThread>yThread)] = err;

        err = (1 - (err_tx_dy / err_tx_ty)) * i; // = ((err_tx_dy / err_tx_ty) - 1) * -i
        if ( ::isnan(err) )
            err = -i * SCALAR_MAX;
        gradient[xThread + NPAIRS*yThread - yThread - (xThread>yThread) + (NPAIRS-1)*(NPAIRS/2)] = err;
    }
}

struct is_positive : public thrust::unary_function<scalar, bool>
{
    __host__ __device__ bool operator()(scalar x){
        return x > 0;
    }
};

extern "C" void profile(int nSamples, int stride, scalar *d_targetParam, double &accuracy, double &median_norm_gradient)
{
    dim3 block(32, 16);
    dim3 grid(NPAIRS/32, NPAIRS/32);
    compute_gradient<<<grid, block>>>(nSamples, stride, d_targetParam, d_gradient);

    thrust::device_ptr<scalar> gradient = thrust::device_pointer_cast(d_gradient);
    thrust::sort(gradient, gradient + gradientSz);

    double nPositive = thrust::count_if(gradient, gradient + gradientSz, is_positive());
    accuracy = nPositive / gradientSz;

    scalar median_g[2];
    CHECK_CUDA_ERRORS(cudaMemcpy(median_g, d_gradient + gradientSz/2, 2*sizeof(scalar), cudaMemcpyDeviceToHost));
    median_norm_gradient = (median_g[0] + median_g[1]) / 2;
}





/// ******************************************************************************************************************************
///  >============================     Elementary Effects WG: Clustering    ====================================================<
/// ******************************************************************************************************************************
static constexpr int STIMS_PER_CLUSTER_BLOCK = 16;
static constexpr int PARTITION_SIZE = 32;

/**
 * @brief build_section_primitives chops the EE traces in d_timeseries into deltabar-normalised deviation vectors ("sections")
 *          representing up to secLen ticks. Sections are chunked into partitions of PARTITION_SIZE=32 sections each.
 *          Deviation vectors represent the mean deviation per tick, normalised to deltabar, caused by a single detuning.
 *          Note, this kernel expects the EE traces to be generated using TIMESERIES_COMPARE_NONE
 * @param out_sections is the output, laid out as [stimIdx][paramIdx][partitionIdx][secIdx (local to partition)].
 * @param out_current is the mean current within each section, laid out as [stimIdx][partitionIdx][secIdx].
 */
__global__ void build_section_primitives(const int trajLen,
                                         const int nTraj,
                                         const int nStims,
                                         const int duration,
                                         const int secLen,
                                         const int nPartitions,
                                         scalar *out_sections,
                                         scalar *out_current)
{
    const int warpid = threadIdx.x / 32; // acts as block-local stim idx
    const int laneid = threadIdx.x & 31;
    const int stimIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // global stim idx; one stim per warp
    const int nTraces = trajLen*nTraj; // Total number of traces per stim, including starting point models
    const int nUsefulTraces = (trajLen-1)*nTraj; // Number of mid-trajectory models per stim
    const int paramIdx_after_end_of_final_traj = nUsefulTraces % NPARAMS; // First param idx with one contrib fewer than the preceding ones
    const int lane0_offset = stimIdx * nTraces;
    const int nLoads = (nTraces + 31) & 0xffffffe0;
    const iObservations obs = dd_obsUNI[stimIdx*nTraces];
    int nextObs = 0;

    volatile __shared__ scalar sh_contrib[STIMS_PER_CLUSTER_BLOCK][NPARAMS][PARTITION_SIZE + 1];
    volatile __shared__ scalar sh_current[STIMS_PER_CLUSTER_BLOCK][PARTITION_SIZE + 1];

    for ( int i = threadIdx.x; i < STIMS_PER_CLUSTER_BLOCK*NPARAMS*(PARTITION_SIZE+1); i += blockDim.x )
        *((scalar*)sh_contrib + i) = 0;
    for ( int i = threadIdx.x; i < STIMS_PER_CLUSTER_BLOCK*(PARTITION_SIZE+1); i += blockDim.x )
        *((scalar*)sh_current + i) = 0;
    __syncthreads();

    if ( stimIdx >= nStims )
        return;

    unsigned int secIdx = 0;
    int trueSecLen_static;
    int t = 0;
    while ( t < duration ) {
        int trueSecLen = 0;
        __syncwarp();
        for ( int tEnd = t + secLen; t < tEnd; t++ ) { // Note, t<duration guaranteed by obs.stop
            if ( nextObs < iObservations::maxObs && t >= obs.start[nextObs] ) {
                if ( t < obs.stop[nextObs] ) {
                    for ( int i = laneid; i < nLoads; i += warpSize ) {
                        const int paramIdx = (i - 1 - (i/trajLen)) % NPARAMS;
                        scalar current_mylane = dd_timeseries[t*NMODELS + lane0_offset + i];
                        scalar current_prevlane = __shfl_up_sync(0xffffffff, current_mylane, 1);
                        scalar diff = scalarfabs(current_prevlane - current_mylane);
                        if ( i < nTraces ) {
                            if ( i % trajLen != 0 )
                                atomicAdd((scalar*)&sh_contrib[warpid][paramIdx][secIdx&31], diff);
                            current_mylane = scalarfabs(current_mylane);
                        } else {
                            current_mylane = 0;
                        }
                        current_mylane = warpReduceSum(current_mylane);
                        if ( laneid == 0 )
                            sh_current[warpid][secIdx&31] += current_mylane;
                    }
                    ++trueSecLen;
                } else {
                    ++nextObs;
                }
            }
        }
        if ( laneid == (secIdx&31) )
            trueSecLen_static = trueSecLen;

        if ( ((++secIdx) & 31) == 0 || t >= duration ) {
            __syncwarp();
            const int partitionIdx = (secIdx-1) >> 5;
            int nContrib = nUsefulTraces/NPARAMS + 1;
            if ( t < duration || laneid <= (secIdx&31) ) {
                for ( int paramIdx = 0; paramIdx < NPARAMS; paramIdx++ ) {
                    if ( paramIdx == paramIdx_after_end_of_final_traj )
                        --nContrib;
                    out_sections[stimIdx * NPARAMS * nPartitions * PARTITION_SIZE
                            + paramIdx * nPartitions * PARTITION_SIZE
                            + partitionIdx * PARTITION_SIZE
                            + laneid]
                            = trueSecLen_static
                              ? sh_contrib[warpid][paramIdx][laneid] / (trueSecLen_static * deltabar[paramIdx] * nContrib)
                              : 0;
                    sh_contrib[warpid][paramIdx][laneid] = 0;
                }
            }

            out_current[stimIdx * nPartitions * PARTITION_SIZE
                    + partitionIdx * PARTITION_SIZE
                    + laneid]
                    = sh_current[warpid][laneid] / (trueSecLen_static * nTraces);
            sh_current[warpid][laneid] = 0;
        }
    }
}

/**
 * @brief compare_within_partition compares all sections in a partition to each other, recording a similarity for each
 * @param myContrib is a section's deviation vector
 * @param dotp_threshold
 * @return a bitmask flagging each above-threshold similar section
 */
__device__ unsigned int compare_within_partition(const Parameters myContrib,
                                                 const scalar norm,
                                                 const scalar dotp_threshold)
{
    const unsigned laneid = threadIdx.x & 31;
    unsigned int mask = 1<<laneid;
    Parameters target_contrib;

    for ( int offset = 1; offset < 17; offset++ ) {
        int target = (laneid + offset)&31;

        // Compare against target
        target_contrib.shfl(myContrib, target);
        scalar target_norm = __shfl_sync(0xffffffff, norm, target);
        scalar dotp = myContrib.dotp(target_contrib);
        if ( dotp > 0 )
            dotp /= (norm * target_norm);

        // Process my own work
        if ( dotp > dotp_threshold ) {
            mask |= 1 << target;
        }

        // Retrieve the work of the thread that targeted me, and process that, too
        target = (laneid + 32 - offset)&31;
        dotp = __shfl_sync(0xffffffff, dotp, target);
        if ( offset < 16 && dotp > dotp_threshold ) {
            mask |= 1 << target;
        }
    }
    return mask;
}

/**
 * @brief compare_within_partition compares all sections in a partition to each other, recording a similarity for each
 * @param myContrib is a section's deviation vector
 * @param dotp_threshold
 * @return a bitmask flagging each above-threshold similar section
 */
__device__ unsigned int compare_between_partitions(const Parameters reference,
                                                   Parameters target,
                                                   const scalar ref_norm,
                                                   const scalar dotp_threshold,
                                                   unsigned int &target_mask)
{
    const unsigned laneid = threadIdx.x & 31;
    unsigned int ref_mask = 0;
    target_mask = 0;
    scalar target_norm = std::sqrt(target.dotp(target));
    const int srcLane = (laneid+1) & 31;
    for ( int i = 0; i < warpSize; i++ ) {
        // Compare against target
        scalar dotp = reference.dotp(target);
        if ( dotp > 0 )
            dotp /= (ref_norm * target_norm);
        if ( dotp > dotp_threshold ) {
            // Update reference
            ref_mask |= 1 << ((laneid+i)&31);

            // Update target
            target_mask |= 1 << laneid;
        }

        // Shuffle targets down (except after the final comparison)
        if ( i < 31 ) {
            target.shfl(target, srcLane);
            target_norm = __shfl_sync(0xffffffff, target_norm, srcLane);
        }
        // shuffle target mask down 32 times to return it to its original lane
        target_mask = __shfl_sync(0xffffffff, target_mask, srcLane);
    }
    return ref_mask;
}

template <typename T>
__device__ unsigned int warpReduceMaxIdx(unsigned int idx, T value)
{
    for ( unsigned int i = 1; i < warpSize; i *= 2 ) {
        T cmp_value = __shfl_down_sync(0xffffffff, value, i);
        unsigned int cmp_idx = __shfl_down_sync(0xffffffff, idx, i);
        if ( cmp_value > value ) {
            value = cmp_value;
            idx = cmp_idx;
        }
    }
    return __shfl_sync(0xffffffff, idx, 0);
}

/**
 * @brief exactClustering is a non-heuristic clustering implementation. It takes the outputs from build_section_primitives, extracting
 * for each stim a set of clusters with associated iObservations, Euclidean normal deviation vector, and mean current.
 */
__global__ void exactClustering(const int nPartitions,
                                const scalar dotp_threshold,
                                const int secLen,
                                const int minClusterLen,
                                scalar *in_contrib, /* [stimIdx][paramIdx][secIdx] */
                                scalar *in_current, /* [stimIdx][secIdx] */
                                scalar *out_clusters, /* [stimIdx][clusterIdx][paramIdx] */
                                scalar *out_clusterCurrent, /* [stimIdx][clusterIdx] */
                                iObservations *out_observations, /* [stimIdx][clusterIdx] */
                                unsigned int *out_masks, /* [stimIdx][partitionIdx][secIdx], intermediate only */
                                const unsigned int shmem_size /* in uints. Minimum nSecs+32, preferably much more for obs timestamp leeway */
                                )
{
    const unsigned laneid = threadIdx.x & 31;
    const unsigned warpid = threadIdx.x >> 5;
    const unsigned int nSecs = 32 * nPartitions;

    extern __shared__ unsigned int shmem[];
    for ( int i = threadIdx.x; i < shmem_size; i += blockDim.x )
        shmem[i] = 0;
    __syncthreads();

    // Part 1: Generate counts and masks
    unsigned int *sh_counts =& shmem[0];
    {
        Parameters reference, target;
        for ( unsigned int refIdx = threadIdx.x; refIdx < nSecs; refIdx += blockDim.x ) {
            reference.load(in_contrib + blockIdx.x * NPARAMS * nSecs + refIdx, nSecs);
            scalar norm = std::sqrt(reference.dotp(reference));
            unsigned int mask = compare_within_partition(reference, norm, dotp_threshold);
            unsigned int count = __popc(mask);
            out_masks[blockIdx.x * nPartitions * nSecs + (refIdx/32) * nSecs + refIdx] = mask;

            for ( int partitionOffset = 1; partitionOffset < nPartitions/2 + 1; partitionOffset++ ) {
                if ( (nPartitions&1) == 0 && partitionOffset == nPartitions/2 && (refIdx/32) >= nPartitions/2 ) {
                    // even # of data sets &&   it's the final iteration      && reference set is in the second half
                    // => This exact comparison has been done and recorded by the first half of reference sets on their final iteration.
                    break;
                }
                const int targetIdx = (refIdx + partitionOffset*32) % nSecs;
                target.load(in_contrib + blockIdx.x * NPARAMS * nSecs + targetIdx, nSecs);
                unsigned int target_mask;
                mask = compare_between_partitions(reference, target, norm, dotp_threshold, target_mask);
                count += __popc(mask);
                atomicAdd(&sh_counts[targetIdx], __popc(target_mask));
                out_masks[blockIdx.x * nPartitions * nSecs + (targetIdx/32) * nSecs + refIdx] = mask;
                out_masks[blockIdx.x * nPartitions * nSecs + (refIdx/32) * nSecs + targetIdx] = target_mask;
            }

            atomicAdd(&sh_counts[refIdx], count);
        }
        __syncthreads();
    }

    // Part 2: Find cluster head indices
    unsigned int static_headIdx;
    unsigned int nClusters;
    {
        unsigned int *sh_headIdx =& shmem[nSecs];
        for ( nClusters = 0; nClusters < MAXCLUSTERS; nClusters++ ) {
            // Block-stride reduction
            unsigned int headIdx = threadIdx.x;
            unsigned int headCount = sh_counts[headIdx];
            for ( unsigned int refIdx = threadIdx.x+blockDim.x; refIdx < nSecs; refIdx += blockDim.x ) {
                unsigned int count = sh_counts[refIdx];
                if ( count > headCount ) {
                    headCount = count;
                    headIdx = refIdx;
                }
            }

            // Warp reduction
            headIdx = warpReduceMaxIdx(headIdx, headCount);
            if ( laneid == 0 )
                sh_headIdx[warpid] = headIdx;
            __syncthreads();

            // Final reduction
            if ( warpid == 0 ) {
                headIdx = sh_headIdx[laneid];
                headCount = headIdx < nSecs ? sh_counts[headIdx] : 0;
                headIdx = warpReduceMaxIdx(headIdx, headCount);

                if ( laneid == 0 ) {
                    // Bail once cluster is too short
                    if ( sh_counts[headIdx] * secLen < minClusterLen )
                        headIdx = nSecs;
                    sh_headIdx[0] = headIdx;
                }
            }
            __syncthreads();

            // Read cluster head
            headIdx = sh_headIdx[0];
            if ( headIdx == nSecs ) // bail
                break;
            if ( threadIdx.x == nClusters )
                static_headIdx = headIdx;

            // Keep followers of head from being heads themselves
            for ( unsigned int secIdx = threadIdx.x; secIdx < nSecs; secIdx += blockDim.x ) {
                if ( out_masks[blockIdx.x * nPartitions * nSecs + (secIdx/32) * nSecs + headIdx] & (1 << (secIdx&31)) )
                    sh_counts[secIdx] = 0;
            }
            __syncthreads();
        }
    }

    // Part 3: Turn the head masks into timestamps
    unsigned int maxStops = shmem_size / nClusters;
    {
        if ( threadIdx.x < nClusters ) {
            unsigned int stopIdx = 1; // starts at 1 to allow space for the final stopIdx at 0
            for ( unsigned int partitionIdx = 0; partitionIdx < nPartitions && stopIdx < maxStops; partitionIdx++ ) {
                unsigned int mask = out_masks[blockIdx.x * nPartitions * nSecs + partitionIdx * nSecs + static_headIdx];
                for ( unsigned int i = 0; i < 32 && stopIdx < maxStops; i++ ) {
                    bool idle = (stopIdx&1); // No observation currently under way
                    bool mustsee = (mask & (1<<i)); // This bit should be included
                    if ( idle == mustsee ) {
                        shmem[threadIdx.x*maxStops + stopIdx] = 32*partitionIdx + i;
                        ++stopIdx;
                    }
                }
            }
            if ( !(stopIdx&1) && stopIdx < maxStops )
                shmem[threadIdx.x*maxStops + stopIdx++] = nPartitions*32-1;
            shmem[threadIdx.x*maxStops] = stopIdx-1;
        }
        __syncthreads();
    }

    // Part 4: Squeeze the timestamps into an iObservations, gather included current & deviations, and store the lot to output
    {
        unsigned int stopIdx;
        for ( unsigned int clusterIdx = warpid; clusterIdx < nClusters; clusterIdx += blockDim.x/32 ) {
            unsigned int nStops = shmem[clusterIdx*maxStops];

            // Shorten as necessary (note: each cluster dealt with in a single warp)
            while ( nStops > 2 * iObservations::maxObs ) {
                unsigned int shortestIdx = 0;
                unsigned int shortestStep = nSecs;
                unsigned int stepLen;

                // warp-stride reduce to find shortest step
                for ( unsigned int i = 0; i < (nStops+30)/31; i++ ) {
                    unsigned int tStop = 0;
                    stopIdx = 31*i + 1 + laneid;
                    if ( stopIdx < nStops )
                        tStop = shmem[clusterIdx*maxStops + stopIdx];
                    stepLen = __shfl_down_sync(0xffffffff, tStop, 1) - tStop;
                    if ( laneid < 31 && stopIdx < nStops && stepLen < shortestStep ) {
                        shortestStep = stepLen;
                        shortestIdx = stopIdx;
                    }
                }
                __syncwarp();

                // final reduce
                for ( unsigned int i = 1; i < warpSize; i *= 2 ) {
                    stepLen = __shfl_down_sync(0xffffffff, shortestStep, i);
                    stopIdx = __shfl_down_sync(0xffffffff, shortestIdx, i);
                    if ( stepLen < shortestStep ) {
                        shortestStep = stepLen;
                        shortestIdx = stopIdx;
                    }
                }
                shortestIdx = __shfl_sync(0xffffffff, shortestIdx, 0);

                // Shift all timestamps from shortestIdx+2 upwards down by two stops to eliminate the identified shorty
                nStops -= 2;
                for ( unsigned int i = shortestIdx/32; i <= nStops/32; i++ ) {
                    unsigned int tmp;
                    unsigned int idx = 32*i + laneid;
                    if ( idx >= shortestIdx && idx <= nStops )
                        tmp = shmem[clusterIdx*maxStops + idx + 2];
                    __syncwarp();
                    if ( idx >= shortestIdx && idx <= nStops )
                        shmem[clusterIdx*maxStops + idx] = tmp;
                }
            }

            // Gather current and deviation values across observed sections
            stopIdx = 0;
            scalar current = 0;
            int nAdditions = 0;
            Parameters contrib, tmp;
            contrib.zero();
            for ( unsigned int secIdx = laneid; secIdx < nSecs; secIdx += warpSize ) {
                while ( stopIdx < nStops && shmem[clusterIdx*maxStops + 1 + stopIdx] <= secIdx )
                    ++stopIdx;
                if ( stopIdx & 1 ) {
                    current += in_current[blockIdx.x * nSecs + secIdx];
                    tmp.load(in_contrib + blockIdx.x * NPARAMS * nSecs + secIdx, nSecs);
                    contrib += tmp;
                    ++nAdditions;
                }
            }
            __syncwarp();

            // Reduce into lane 0
            for ( unsigned int i = 1; i < warpSize; i *= 2 ) {
                current += __shfl_down_sync(0xffffffff, current, i);
                nAdditions += __shfl_down_sync(0xffffffff, nAdditions, i);
                tmp.shfl(contrib, laneid + i);
                contrib += tmp;
            }

            // Store output
            if ( laneid == 0 ) {
                iObservations obs = {{}, {}};
                for ( unsigned int i = 0; i < nStops/2; i++ ) {
                    obs.start[i] = shmem[clusterIdx*maxStops + 2*i + 1] * secLen;
                    obs.stop[i] = shmem[clusterIdx*maxStops + 2*i + 2] * secLen;
                }
                out_observations[blockIdx.x * MAXCLUSTERS + clusterIdx] = obs;

                contrib /= std::sqrt(contrib.dotp(contrib));
                contrib.store(out_clusters + blockIdx.x * MAXCLUSTERS * NPARAMS + clusterIdx * NPARAMS);

                current /= nAdditions;
                out_clusterCurrent[blockIdx.x * MAXCLUSTERS + clusterIdx] = current;
            }
        }

        // Backstop
        if ( nClusters < MAXCLUSTERS && threadIdx.x == 0 ) {
            out_observations[blockIdx.x * MAXCLUSTERS + nClusters] = iObservations {{},{}};
        }
    }
}

extern "C" void pullClusters(int nStims)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nStims * MAXCLUSTERS * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterCurrent, d_clusterCurrent, nStims * MAXCLUSTERS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterObs, d_clusterObs, nStims * MAXCLUSTERS * sizeof(iObservations), cudaMemcpyDeviceToHost));
}

extern "C" int pullPrimitives(int nStims, int duration, int secLen)
{
    int nSecs = (duration+secLen-1)/secLen;
    int nPartitions = (nSecs + 31)/32;
    CHECK_CUDA_ERRORS(cudaMemcpy(sections, d_sections, nStims * nPartitions * NPARAMS * PARTITION_SIZE * sizeof(scalar), cudaMemcpyDeviceToHost));
    return nPartitions * PARTITION_SIZE;
}

extern "C" void cluster(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                       int nTraj, /* Number of EE trajectories */
                       int duration,
                       int secLen,
                       scalar dotp_threshold,
                       int minClusterLen,
                       std::vector<double> deltabar_arg,
                       bool pull_results)
{
    unsigned int nStims = NMODELS / (trajLen*nTraj);
    unsigned int nClusters = nStims * MAXCLUSTERS;
    int nSecs = (duration+secLen-1)/secLen;
    int nPartitions = (nSecs + 31)/32;

    resizeArrayPair(sections, d_sections, sections_size, nStims * nPartitions * NPARAMS * PARTITION_SIZE);
    resizeArray(d_currents, currents_size, nStims * nPartitions * PARTITION_SIZE);
    resizeArrayPair(clusters, d_clusters, clusters_size, nClusters * NPARAMS);
    resizeArray(d_clusterMasks, clusterMasks_size, nStims * nPartitions * 32*nPartitions);
    resizeArrayPair(clusterCurrent, d_clusterCurrent, clusterCurrent_size, nClusters);
    resizeArrayPair(clusterObs, d_clusterObs, clusterObs_size, nClusters);

    scalar deltabar_array[NPARAMS];
    for ( int i = 0; i < NPARAMS; i++ )
        deltabar_array[i] = deltabar_arg[i];
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(deltabar, deltabar_array, NPARAMS*sizeof(scalar)));

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    build_section_primitives<<<grid, block>>>(trajLen, nTraj, nStims, duration, secLen, nPartitions, d_sections, d_currents);

    size_t shmem_size = std::max(32*nPartitions, 8192);
    size_t nWarps = 16;
    exactClustering<<<nStims, 32*nWarps, shmem_size*sizeof(int)>>>(nPartitions, dotp_threshold, secLen, minClusterLen,
                                                                   d_sections, d_currents,
                                                                   d_clusters, d_clusterCurrent, d_clusterObs,
                                                                   d_clusterMasks, shmem_size);


    if ( pull_results )
        pullClusters(nStims);
}






/// ******************************************************************************************************************************
///  >============================     Elementary Effects WG: Bubbles    =======================================================<
/// ******************************************************************************************************************************

__device__ inline Bubble bubble_shfl(Bubble b, int srcLane)
{
    return Bubble {
        __shfl_sync(0xffffffff, b.startCycle, srcLane),
        __shfl_sync(0xffffffff, b.cycles, srcLane),
        __shfl_sync(0xffffffff, b.value, srcLane)
    };
}

/**
 * @brief warpMergeBubbles merges bubble triplets across a warp, storing in lane 0's bubbles the fully merged triplet
 * @param cyclesPerTriplet Number of cycles (sections?) that each triplet summarises
 * @param cutoff Number of threads participating in the merge (power of 2, or pass zero for other threads' triplets. All threads must call this function in sync.)
 */
__device__ void warpMergeBubbles(Bubble &start, Bubble &mid, Bubble &end, int cyclesPerTriplet, int cutoff)
{
    Bubble nextStart, nextMid, nextEnd;
    const unsigned int laneid = threadIdx.x&31;
    for ( int offset = 1; offset < cutoff; offset *= 2 ) {
        // Shuffle down
        nextStart = bubble_shfl(start, laneid+offset);
        nextMid = bubble_shfl(mid, laneid+offset);
        nextEnd = bubble_shfl(end, laneid+offset);

        // Merge across the seam
        if ( nextStart.cycles ) {
            if ( start.cycles == cyclesPerTriplet*offset ) { // full-length merge
                // Merge nextStart into start; invalidate nextStart
                start.cycles += nextStart.cycles;
                start.value += nextStart.value;
                nextStart.cycles = 0;
            } else if ( end.cycles ) { // end merge
                if ( nextStart.cycles == cyclesPerTriplet*offset ) { // nextStart is full-length
                    // Merge end and nextStart, save to new end, invalidate end and nextStart
                    nextEnd = {end.startCycle,
                               end.cycles+nextStart.cycles,
                               end.value+nextStart.value};
                    end.cycles = 0;
                } else {
                    // Merge nextStart into end; invalidate nextStart
                    end.cycles += nextStart.cycles;
                    end.value += nextStart.value;
                }
                nextStart.cycles = 0;
            } else if ( nextStart.cycles == cyclesPerTriplet*offset ) { // No merge, nextStart is new end
                nextEnd = nextStart;
                nextStart.cycles = 0;
            }
        }

        // Replace mid with the largest of (mid, end, nextStart, nextMid)
        scalar midFitness = mid.cycles ? mid.value/mid.cycles : 0;
        if ( end.cycles && end.value/end.cycles > midFitness ) {
            midFitness = end.value/end.cycles;
            mid = end;
        }
        if ( nextStart.cycles && nextStart.value/nextStart.cycles > midFitness ) {
            midFitness = nextStart.value/nextStart.cycles;
            mid = nextStart;
        }
        if ( nextMid.cycles && nextMid.value/nextMid.cycles > midFitness )
            mid = nextMid;

        // Replace end with nextEnd
        end = nextEnd;
    }
}

__global__ void buildBubbles(const int nPartitions,
                             const int secLen,
                             scalar *in_contrib, /* [stimIdx][paramIdx][secIdx] */
                             scalar *in_current, /* [stimIdx][secIdx] */
                             scalar *out_deviations, /* [stimIdx][targetParamIdx][paramIdx] */
                             scalar *out_bubbleCurrents, /* [stimIdx][targetParamIdx] */
                             Bubble *out_bubbles /* [stimIdx][targetParamIdx] */
                             )
{
    const unsigned int nSecs = 32*nPartitions;
    const unsigned int warpid = threadIdx.x/32;
    const unsigned int laneid = threadIdx.x&31;
    const unsigned int nWarps = blockDim.x/32;
    const unsigned int stimIdx = blockIdx.x;
    const unsigned int targetParamIdx = blockIdx.y;

    Bubble start, mid, end;
    Parameters dev;
    extern __shared__ unsigned int shmem[];
    Bubble *sh_start = (Bubble*) &shmem[0];
    Bubble *sh_mid =& sh_start[nPartitions];
    Bubble *sh_end =& sh_mid[nPartitions];

    // Generate bubble triplets over partitions
    for ( unsigned int partitionIdx = warpid; partitionIdx < nPartitions; partitionIdx += nWarps ) {
        const unsigned int secIdx = partitionIdx*32 + laneid;
        // Generate initial bubble triplet over section
        dev.load(in_contrib + stimIdx * NPARAMS * nSecs + secIdx, nSecs);
        start.value = dev[targetParamIdx] / dev.mean();
        start.startCycle = secIdx;
        start.cycles = (start.value > 1);
        mid = {0,0,0};
        end = {0,0,0};

        // Merge to lane 0
        warpMergeBubbles(start, mid, end, 1, warpSize);

        if ( laneid == 0 ) {
            sh_start[partitionIdx] = start;
            sh_mid[partitionIdx] = mid;
            sh_end[partitionIdx] = end;
        }
    }

    __syncthreads();

    // Reduce across partitions in warp 0
    if ( warpid == 0 ) {
        Bubble prevStart, prevMid, prevEnd;
        for ( unsigned int metaPartitionIdx = 0; metaPartitionIdx < (nPartitions+31)/32; metaPartitionIdx++ ) {
            const unsigned int partitionIdx = metaPartitionIdx*32 + laneid;
            // Load partition triplet
            if ( partitionIdx < nPartitions ) {
                start = sh_start[partitionIdx];
                mid = sh_mid[partitionIdx];
                end = sh_end[partitionIdx];
            } else {
                start = mid = end = {0,0,0};
            }

            // Merge and broadcast
            warpMergeBubbles(start, mid, end, warpSize, warpSize);
            start = bubble_shfl(start, 0);
            mid = bubble_shfl(mid, 0);
            end = bubble_shfl(end, 0);

            // Merge the previous MetaPartition's triplet in lane 0 with the current one in lane 1
            if ( metaPartitionIdx > 0 ) {
                if ( laneid == 0 ) {
                    start = prevStart;
                    mid = prevMid;
                    end = prevEnd;
                }
                warpMergeBubbles(start, mid, end, 1024, 2);
            }
            if ( laneid == 0 ) {
                prevStart = start;
                prevMid = mid;
                prevEnd = end;
            }
        }

        // Pick the winner, store to shmem and to output
        if ( laneid == 0 ) {
            // Assume mid to be the fittest (likeliest branch)
            scalar fitness = mid.cycles ? mid.value/mid.cycles : 0;
            if ( start.cycles && start.value/start.cycles > fitness ) {
                fitness = start.value/start.cycles;
                mid = start;
            }
            if ( end.cycles && end.value/end.cycles > fitness ) {
                fitness = end.value/end.cycles;
                mid = end;
            }
            mid.value = fitness;
            sh_start[0] = mid;

            // Adjust timings from secs to ticks
            mid.cycles *= secLen;
            mid.startCycle *= secLen;
            out_bubbles[stimIdx * NPARAMS + targetParamIdx] = mid;
        }
    }

    __syncthreads();
    start = sh_start[0];
    __syncthreads();

    // Discard use of shmem as Bubble[], use it for current/deviation reduction instead
    // Note the size requirement for 32 scalars and 32 Parameters.
    scalar *sh_current = (scalar*)&shmem[0];
    Parameters *sh_deviation = (Parameters*)&sh_current[32];
    if ( laneid == 0 ) {
        sh_current[warpid] = 0;
        sh_deviation[warpid].zero();
    }

    // Gather bubble deviation and current
    for ( unsigned int secIdx = start.startCycle, lastSec = start.startCycle + start.cycles; secIdx < (lastSec+31)&0xffffffe0; secIdx += blockDim.x ) {
        scalar current;
        if ( secIdx < lastSec ) {
            dev.load(in_contrib + stimIdx * NPARAMS * nSecs + secIdx, nSecs);
            current = in_current[stimIdx * nSecs + secIdx];
        } else {
            dev.zero();
            current = 0;
        }

        current = warpReduceSum(current);
        dev = warpReduceSum(dev);

        if ( laneid == 0 ) {
            sh_current[warpid] += current;
            sh_deviation[warpid] += dev;
        }
    }

    __syncthreads();

    if ( warpid == 0 ) {
        scalar current = warpReduceSum(sh_current[laneid]);
        dev = warpReduceSum(sh_deviation[laneid]);
        if ( laneid == 0 ) {
            out_bubbleCurrents[stimIdx * NPARAMS + targetParamIdx] = current;
            dev /= std::sqrt(dev.dotp(dev));
            dev.store(out_deviations + stimIdx * NPARAMS * NPARAMS + targetParamIdx * NPARAMS);
        }
    }
}

extern "C" void pullBubbles(int nStims)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(bubbles, d_bubbles, nStims * NPARAMS * sizeof(Bubble), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterCurrent, d_clusterCurrent, nStims * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nStims * NPARAMS * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
}

extern "C" void bubble(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                       int nTraj, /* Number of EE trajectories */
                       int duration,
                       int secLen,
                       std::vector<double> deltabar_arg,
                       bool pull_results)
{
    unsigned int nStims = NMODELS / (trajLen*nTraj);
    int nSecs = (duration+secLen-1)/secLen;
    int nPartitions = (nSecs + 31)/32;

    resizeArrayPair(sections, d_sections, sections_size, nStims * nPartitions * NPARAMS * PARTITION_SIZE);
    resizeArray(d_currents, currents_size, nStims * nPartitions * PARTITION_SIZE);
    resizeArrayPair(clusters, d_clusters, clusters_size, nStims * NPARAMS * NPARAMS);
    resizeArrayPair(clusterCurrent, d_clusterCurrent, clusterCurrent_size, nStims * NPARAMS);
    resizeArrayPair(bubbles, d_bubbles, bubbles_size, nStims * NPARAMS);

    scalar deltabar_array[NPARAMS];
    for ( int i = 0; i < NPARAMS; i++ )
        deltabar_array[i] = deltabar_arg[i];
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(deltabar, deltabar_array, NPARAMS*sizeof(scalar)));

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    build_section_primitives<<<grid, block>>>(trajLen, nTraj, nStims, duration, secLen, nPartitions, d_sections, d_currents);

    size_t nWarps = 16;
    size_t shmem_for_bubbles = 3 * nPartitions * sizeof(Bubble);
    size_t shmem_for_stats = 32 * sizeof(scalar) + 32 * sizeof(Parameters);
    size_t shmem_size = std::max(shmem_for_bubbles, shmem_for_stats);
    block = dim3(nWarps * 32);
    grid = dim3(nStims, NPARAMS);
    buildBubbles<<<grid, block, shmem_size>>>(nPartitions, secLen, d_sections, d_currents, d_clusters, d_clusterCurrent, d_bubbles);

    if ( pull_results )
        pullBubbles(nStims);
}





/// ******************************************************************************************************************************
///  >============================     Elementary Effects WG: Deltabar    ======================================================<
/// ******************************************************************************************************************************

static constexpr int STIMS_PER_DELTABAR_WARP = NPARAMS>16 ? 1 : NPARAMS>8 ? 2 : NPARAMS>4 ? 4 : NPARAMS>2 ? 8 : NPARAMS>1 ? 16 : 32;
static constexpr int STIMS_PER_DELTABAR_BLOCK = 8 * STIMS_PER_DELTABAR_WARP;

__global__ void find_deltabar_kernel(int trajLen, int nTraj, int nStims, int duration, scalar *global_clusters, int *global_clusterLen)
{
    static constexpr int stimWidth = 32/STIMS_PER_DELTABAR_WARP;
    const int nUsefulTraces = (trajLen-1)*nTraj;
    int paramIdx, stimIdx;
    if ( STIMS_PER_DELTABAR_WARP == 1 ) {
        // One row, one stim
        paramIdx = threadIdx.x;
        stimIdx = threadIdx.y;
    } else {
        // Each row is one warp with multiple stims
        paramIdx = threadIdx.x % stimWidth;
        stimIdx = (threadIdx.y * STIMS_PER_DELTABAR_WARP) + (threadIdx.x/stimWidth);
    }
    const int global_stimIdx = STIMS_PER_DELTABAR_BLOCK*blockIdx.x + stimIdx;
    const int id0 = global_stimIdx * trajLen * nTraj;
    const int nContrib = (nUsefulTraces/NPARAMS) + (paramIdx < (nUsefulTraces%NPARAMS));
    const iObservations obs = dd_obsUNI[id0];
    int nextObs = 0;
    int nSamples = 0;

    __shared__ scalar sh_clusters[STIMS_PER_DELTABAR_BLOCK][NPARAMS];
    __shared__ scalar sh_nSamples[STIMS_PER_DELTABAR_BLOCK];
    const int tid = threadIdx.y*blockDim.x + threadIdx.x;

    // Accumulate each stim's square deviations
    scalar sumSquares = 0;
    if ( paramIdx < NPARAMS ) {
        if ( global_stimIdx < nStims ) {
            for ( int t = 0; t < duration; t++ ) {
                if ( nextObs < iObservations::maxObs && t >= obs.start[nextObs] ) {
                    if ( t < obs.stop[nextObs] ) {
                        scalar contrib = 0;
                        for ( int i = paramIdx; i < nUsefulTraces; i += NPARAMS ) {
                            contrib += scalarfabs(dd_timeseries[t*NMODELS + id0 + i + i/(trajLen-1) + 1]);
                        }
                        contrib /= nContrib;
                        sumSquares += contrib*contrib;
                        ++nSamples;
                    } else {
                        ++nextObs;
                    }
                }
            }
        }
        sh_clusters[stimIdx][paramIdx] = sumSquares;
        if ( paramIdx == 0 )
            sh_nSamples[stimIdx] = nSamples;
    }

    // Reduce to a single 'cluster' in block
    for ( int width = STIMS_PER_DELTABAR_BLOCK; width > 1; width /= 32 ) {
        paramIdx = tid / width;
        stimIdx = tid % width;
        sumSquares = 0;
        nSamples = 0;
        __syncthreads();
        if ( paramIdx < NPARAMS ) {
            sumSquares = sh_clusters[stimIdx][paramIdx];
            if ( paramIdx == 0 )
                nSamples = sh_nSamples[stimIdx];
        }

        if ( width > 32 ) {
            sumSquares = warpReduceSum(sumSquares);
            if ( paramIdx == 0 )
                nSamples = warpReduceSum(nSamples);
            if ( stimIdx % 32 == 0 ) {
                sh_clusters[stimIdx/32][paramIdx] = sumSquares;
                if ( paramIdx == 0 )
                    sh_nSamples[stimIdx/32] = nSamples;
            }
        } else {
            sumSquares = warpReduceSum(sumSquares, width);
            if ( paramIdx == 0 )
                nSamples = warpReduceSum(nSamples, width);
        }
    }
    if ( stimIdx == 0 && paramIdx < NPARAMS ) {
        global_clusters[blockIdx.x*NPARAMS + paramIdx] = sumSquares;
        if ( paramIdx == 0 )
            global_clusterLen[blockIdx.x] = nSamples;
    }
}

extern "C" std::vector<double> find_deltabar(int trajLen, int nTraj, int duration)
{
    unsigned int nStims = NMODELS / (trajLen*nTraj);
    dim3 block(((NPARAMS+31)/32)*32, STIMS_PER_DELTABAR_BLOCK/STIMS_PER_DELTABAR_WARP);
    dim3 grid(((nStims+STIMS_PER_DELTABAR_BLOCK-1)/STIMS_PER_DELTABAR_BLOCK));

    resizeArrayPair(clusters, d_clusters, clusters_size, grid.x * NPARAMS);
    resizeArrayPair(clusterLen, d_clusterLen, clusterLen_size, grid.x);

    find_deltabar_kernel<<<grid, block>>>(trajLen, nTraj, nStims, duration, d_clusters, d_clusterLen);

    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, grid.x * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterLen, d_clusterLen, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

    // Reduce across blocks on the CPU - this isn't performance-critical
    int n = 0;
    std::vector<double> ret(NPARAMS, 0);
    for ( int i = 0; i < grid.x; i++ ) {
        for ( int p = 0; p < NPARAMS; p++ )
            ret[p] += clusters[i*NPARAMS + p];
        n += clusterLen[i];
    }
    for ( int p = 0; p < NPARAMS; p++ )
        ret[p] = sqrt(ret[p] / n);
    return ret;
}





/// ******************************************************************************************************************************
///  >============================     Utility functions    ====================================================================<
/// ******************************************************************************************************************************

__global__ void observe_no_steps_kernel(int blankCycles)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if ( id >= NMODELS )
        return;
    iStimulation stim = dd_stimUNI[id];
    iObservations obs = {};
    int tStart = 0;
    int nextObs = 0;
    for ( const auto step : stim ) {
        if ( step.t > stim.duration )
            break;
        if ( !step.ramp ) {
            if ( tStart < step.t ) {
                obs.start[nextObs] = tStart;
                obs.stop[nextObs] = step.t;
                if ( ++nextObs == iObservations::maxObs )
                    break;
            }
            tStart = step.t + blankCycles;
        }
    }
    if ( nextObs < iObservations::maxObs ) {
        if ( tStart < stim.duration ) {
            obs.start[nextObs] = tStart;
            obs.stop[nextObs] = stim.duration;
        }
    }
    dd_obsUNI[id] = obs;
}

extern "C" void observe_no_steps(int blankCycles)
{
    dim3 block(256);
    observe_no_steps_kernel<<<((NMODELS+block.x-1)/block.x)*block.x, block.x>>>(blankCycles);
}

#endif
