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

static scalar *clusterCurrent = nullptr, *d_clusterCurrent = nullptr;
static unsigned int clusterCurrent_size = 0;

static scalar *sections = nullptr, *d_sections = nullptr;
static unsigned int sections_size = 0;

static scalar *d_currents= nullptr;
static unsigned int currents_size = 0;

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
    pointers.clusterLen =& clusterLen;
    pointers.clusterCurrent =& clusterCurrent;
    pointers.clusterPrimitives =& sections;

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


// Code adapated from Justin Luitjens, <https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/>
// Note, shuffle is only supported on compute capability 3.x and higher
__device__ inline scalar warpReduceSum(scalar val, int cutoff = warpSize)
{
    for ( int offset = 1; offset < cutoff; offset *= 2 )
        val += __shfl_xor_sync(0xffffffff, val, offset);
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
                        scalar diff = current_prevlane - current_mylane;
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
 * @brief find_single_section_representative finds the best representative section within a warp-sized partition,
 *      where "best" means "greatest number of followers", and ties are broken in favour of a higher mean dotp to follower sections
 * @return ((maxFollowers<<5) | rep_laneid) to all threads
 */
__device__ inline unsigned int find_single_section_representative(const unsigned int nFollowers,
                                                                  const unsigned int laneid,
                                                                  scalar mean_dotp)
{
    // Pack laneid and nFollowers into a single int (they're both <= 32)
    unsigned int max_count_and_lane = (nFollowers<<5) | laneid;
    for ( unsigned int delta = 1; delta < warpSize; delta <<= 1 ) {
        const unsigned int target = __shfl_xor_sync(0xffffffff, max_count_and_lane, delta);
        const unsigned int target_nFollowers = target>>5;
        const scalar target_dotp = __shfl_xor_sync(0xffffffff, mean_dotp, delta);
        if ( (target_nFollowers > nFollowers)
             || (target_nFollowers == nFollowers && target_dotp > mean_dotp)
             || (target_nFollowers == nFollowers && target_dotp == mean_dotp && delta < laneid) ) {
            max_count_and_lane = target;
            mean_dotp = target_dotp;
        }
    }

    return max_count_and_lane;
}

/**
 * @brief warp_compare_sections_all2all compares all sections in a warp to each other, recording similarity and the sum of above-threshold dotp's
 * @param myContrib is a section's deviation vector
 * @param dotp_threshold
 * @param sum_dotp is increased by the sum of above-threshold dotp's
 * @return a bitmask flagging each above-threshold similar section
 */
__device__ unsigned int warp_compare_sections_all2all(const Parameters myContrib,
                                                      const scalar dotp_threshold,
                                                      scalar &sum_dotp)
{
    const unsigned laneid = threadIdx.x & 31;
    unsigned int mask = 1<<laneid;
    scalar norm = std::sqrt(myContrib.dotp(myContrib));
    Parameters target_contrib;

    for ( unsigned int offset = 1; offset < 17; offset++ ) {
        int target = (laneid + offset)&31;

        // Compare against target
        target_contrib.shfl(myContrib, target);
        scalar target_norm = __shfl_sync(0xffffffff, norm, target);
        scalar dotp = scalarfabs(myContrib.dotp(target_contrib));
        if ( dotp > 0 ) // ... else, one of the norms is zero, too
            dotp /= (norm * target_norm);

        // Process my own work
        if ( dotp > dotp_threshold ) {
            sum_dotp += dotp;
            mask |= 1 << target;
        }

        // Retrieve the work of the thread that targeted me, and process that, too
        target = (laneid + 32 - offset)&31;
        dotp = __shfl_sync(0xffffffff, dotp, target);
        if ( offset < 16 && dotp > dotp_threshold ) {
            sum_dotp += dotp;
            mask |= 1 << target;
        }
    }
    return mask;
}

/**
 * @brief warp_compare_reps_all2all compares all representative sections in a warp to each other, recording similarity and the total estimated follower count for each rep
 * @param myContrib is a rep's deviation vector
 * @param dotp_threshold
 * @param nOwnFollowers is the number of followers of the rep in myContrib within its own partition
 * @param nEstimatedFollowers is increased by the number of own followers of each rep that is similar to this thread's, EXCLUDING self.
 * @return a bitmask flagging each above-threshold similar rep
 */
__device__ unsigned int warp_compare_reps_all2all(const Parameters myContrib,
                                                  const scalar norm,
                                                  const scalar dotp_threshold,
                                                  const int nOwnFollowers,
                                                  int &nEstimatedFollowers)
{
    const unsigned laneid = threadIdx.x & 31;
    unsigned int mask = 1<<laneid;
    Parameters target_contrib;

    for ( int offset = 1; offset < 17; offset++ ) {
        int target = (laneid + offset)&31;

        // Compare against target
        target_contrib.shfl(myContrib, target);
        scalar target_norm = __shfl_sync(0xffffffff, norm, target);
        int target_nOwnFollowers = __shfl_sync(0xffffffff, nOwnFollowers, target);
        scalar dotp = scalarfabs(myContrib.dotp(target_contrib));
        if ( dotp > 0 )
            dotp /= (norm * target_norm);

        // Process my own work
        if ( dotp > dotp_threshold ) {
            nEstimatedFollowers += target_nOwnFollowers;
            mask |= 1 << target;
        }

        // Retrieve the work of the thread that targeted me, and process that, too
        target = (laneid + 32 - offset)&31;
        dotp = __shfl_sync(0xffffffff, dotp, target);
        target_nOwnFollowers = __shfl_sync(0xffffffff, nOwnFollowers, target);
        if ( offset < 16 && dotp > dotp_threshold ) {
            nEstimatedFollowers += target_nOwnFollowers;
            mask |= 1 << target;
        }
    }
    return mask;
}

/**
 * @brief find_section_representatives extracts the two most representative sections from a partition (=32 sections), such that
 *          neither representative is above-threshold similar to (= a follower of) the other
 * @param nSecs total number of sections in the input
 * @param nPartitions total number of 32-section partitions
 * @param in_contrib Input data (deviation vectors); ordered by [stimIdx][paramIdx][partitionIdx][sectionIdx]
 * @param out_reps Copy of the two representative section's deviation vectors; ordered by [repIdx][paramIdx],
 *           where repIdx is 2*partitionIdx for the primary and 2*partitionIdx+1 for the secondary representative
 * @param out_ownFollowerMasks Bitmasks for each rep flagging its followers, including self; ordered by [repIdx]
 */
__device__ void find_section_representatives(const int nSecs,
                                             const int nPartitions,
                                             const scalar dotp_threshold,
                                             scalar *in_contrib,
                                             scalar *out_reps,
                                             unsigned int *out_ownFollowerMasks)
{
    const unsigned int laneid = threadIdx.x & 31;
    const unsigned int warpid = threadIdx.x >> 5;
    Parameters myContrib;

    for ( int partitionIdx = warpid; partitionIdx < nPartitions; partitionIdx += blockDim.x>>5 ) {
        if ( partitionIdx*PARTITION_SIZE + laneid < nSecs)
            myContrib.load(in_contrib
                           + PARTITION_SIZE * nPartitions * NPARAMS * blockIdx.x
                           /* no indexing for parameter: that's done in .load() using the stride argument */
                           + PARTITION_SIZE * partitionIdx
                           + laneid,
                    PARTITION_SIZE * nPartitions);
        else
            myContrib.zero();
        __syncwarp();

        // All-to-all comparison within warp/partition
        scalar sum_dotp = 0;
        unsigned int similar_lanes = warp_compare_sections_all2all(myContrib, dotp_threshold, sum_dotp);

        // Find primary representative (greatest number of similar lanes, ties broken with greatest sum dotp)
        unsigned int max_count_and_lane = find_single_section_representative(__popc(similar_lanes), laneid, sum_dotp);

        // Store primary representative, provided it has at least one follower
        if ( laneid == (max_count_and_lane&31) && (max_count_and_lane>>6) ) {
            myContrib.store(out_reps + 2*partitionIdx * NPARAMS);
            out_ownFollowerMasks[2*partitionIdx] = similar_lanes;
        }

        // Find a secondary representative if there are at least two unrepresented sections remaining
        if ( max_count_and_lane < (31<<5) ) {
            // Remove the primary rep's followers' ability to represent
            if ( (1<<laneid) & __shfl_sync(0xffffffff, similar_lanes, max_count_and_lane&31) )
                similar_lanes = 0;

            // Find secondary representative. Since followers of the primary are only excluded from being reps, but not from being
            // the secondary rep's followers, sum_dotp remains a valid tie breaker for the remaining potential reps.
            max_count_and_lane = find_single_section_representative(__popc(similar_lanes), laneid, sum_dotp);

            // Store secondary representative
            if ( laneid == (max_count_and_lane&31) && (max_count_and_lane>>6) ) {
                myContrib.store(out_reps + (2*partitionIdx + 1) * NPARAMS);
                out_ownFollowerMasks[2*partitionIdx + 1] = similar_lanes;
            }
        }
    }
}

/**
 * @brief metaPartition_compare_reps_all2all compares the given reference reps against reps in all metapartitions EXCEPT the reference's own
 * @param myContrib (reference)
 * @param norm      (reference)
 * @param repIdx    (reference)
 * @param out_nEstimatedFollowers   Output: Target reps' estimated follower count is atomically updated to include the relevant reference reps' own followers
 * @param out_repMasks  Output: Follower bitmasks for both reference and targets
 * @return the reference rep's estimated follower count outside of its own metapartition
 */
__device__ int metaPartition_compare_reps_all2all(const Parameters myContrib,
                                                  const scalar norm,
                                                  const unsigned int repIdx,
                                                  const unsigned int nReps,
                                                  const unsigned int nMetaPartitions,
                                                  const scalar dotp_threshold,
                                                  scalar *in_reps,
                                                  unsigned int *in_ownFollowerMasks,
                                                  unsigned int *out_nEstimatedFollowers,
                                                  unsigned int *out_repMasks)
{
    Parameters target_contrib;
    const unsigned laneid = threadIdx.x & 31;
    scalar target_norm;
    int nOwnFollowers = 0, nEstimatedFollowers = 0;
    int target_nOwnFollowers, target_nEstimatedFollowers;
    unsigned int repMask, target_repMask;

    for ( int warpTargetOffset = 1; warpTargetOffset < nMetaPartitions/2 + 1; warpTargetOffset++ ) {
        if ( (nMetaPartitions&1) == 0 && warpTargetOffset == nMetaPartitions/2 && (repIdx>>5) >= nMetaPartitions/2 ) {
            // even # of data sets  &&   it's the final iteration           &&   reference set is in the second half
            // => This exact comparison has been done and recorded by the first half of reference sets on their final iteration.
            break;
        }

        const int targetRepIdx = (repIdx + warpTargetOffset*32) % (nMetaPartitions*32); // look ahead by wTO sets of 32
        if ( targetRepIdx < nReps ) {
            target_contrib.load(in_reps + targetRepIdx * NPARAMS);
            target_norm = std::sqrt(target_contrib.dotp(target_contrib));
            target_nOwnFollowers = __popc(in_ownFollowerMasks[targetRepIdx]);
        } else {
            target_contrib.zero();
            target_norm = 0;
            target_nOwnFollowers = 0;
        }
        target_nEstimatedFollowers = 0;
        repMask = target_repMask = 0;
        __syncwarp();

        // Compare every reference against every target (32 comparisons per thread) by rotating targets by one thread at a time
        const int srcLane = (laneid+1) & 31;
        for ( int i = 0; i < 32; i++ ) {
            // Compare against target
            scalar dotp = scalarfabs(myContrib.dotp(target_contrib));
            if ( dotp > 0 )
                dotp /= (norm * target_norm);
            if ( dotp > dotp_threshold ) {
                // Update reference
                nEstimatedFollowers += target_nOwnFollowers;
                repMask |= 1 << ((laneid+i)&31);

                // Update target
                target_nEstimatedFollowers += nOwnFollowers;
                target_repMask |= 1 << laneid;
            }

            // Shuffle targets down (except after the final comparison)
            if ( i < 31 ) {
                target_contrib.shfl(target_contrib, srcLane);
                target_norm = __shfl_sync(0xffffffff, target_norm, srcLane);
                target_nOwnFollowers = __shfl_sync(0xffffffff, target_nOwnFollowers, srcLane);
            }
            // shuffle target outputs down 32 times to return them to their original lane
            target_nEstimatedFollowers = __shfl_sync(0xffffffff, target_nEstimatedFollowers, srcLane);
            target_repMask = __shfl_sync(0xffffffff, target_repMask, srcLane);
        }

        // Store target outputs back into shared
        if ( repIdx < nReps ) {
            if ( targetRepIdx < nReps ) {
                atomicAdd(out_nEstimatedFollowers + targetRepIdx, target_nEstimatedFollowers);
                out_repMasks[(repIdx>>5) * nReps + targetRepIdx] = target_repMask;
            }
            out_repMasks[(targetRepIdx>>5) * nReps + repIdx] = repMask;
        }
    }

    return nEstimatedFollowers;
}

/**
 * @brief estimate_rep_follower_count compares reps all2all, accumulating the followers of similar reps into nEstimatedFollowers
 * @param out_repMasks Similarity bitmasks arranged as [metaPartitionIdx][repIdx]
 */
__device__ void estimate_rep_follower_count(const unsigned int nReps,
                                            const unsigned int nMetaPartitions,
                                            const scalar dotp_threshold,
                                            scalar *in_reps,
                                            unsigned int *in_ownFollowerMasks,
                                            unsigned int *out_nEstimatedFollowers,
                                            unsigned int *out_repMasks)
{
    Parameters myContrib;
    const unsigned laneid = threadIdx.x & 31;
    const unsigned warpid = threadIdx.x >> 5;

    for ( unsigned int referenceWarpIdx = warpid; referenceWarpIdx < nMetaPartitions; referenceWarpIdx += blockDim.x>>5 ) {
        const unsigned int repIdx = (referenceWarpIdx<<5) + laneid;
        int nOwnFollowers = 0;
        if ( repIdx < nReps ) {
            // Load the reference reps
            myContrib.load(in_reps + repIdx * NPARAMS);
            nOwnFollowers = __popc(in_ownFollowerMasks[repIdx]);
        } else {
            myContrib.zero();
        }
        scalar norm = std::sqrt(myContrib.dotp(myContrib));
        __syncwarp();

        // Compare amongst the reference reps (all-to-all within metapartition)
        int nEstimatedFollowers = nOwnFollowers;
        unsigned int repMask = warp_compare_reps_all2all(myContrib, norm, dotp_threshold, nOwnFollowers, nEstimatedFollowers);
        if ( repIdx < nReps )
            out_repMasks[(repIdx>>5) * nReps + repIdx] = repMask;

        // Compare against all other metapartitions
        nEstimatedFollowers += metaPartition_compare_reps_all2all(myContrib, norm, repIdx, nReps, nMetaPartitions, dotp_threshold,
                                                                  in_reps, in_ownFollowerMasks,
                                                                  out_nEstimatedFollowers, out_repMasks);

        // Add forward estimate to output
        if ( repIdx < nReps )
            atomicAdd(out_nEstimatedFollowers + repIdx, nEstimatedFollowers);
    }
}

/**
 * @brief block_extract_clusters finds at most MAXCLUSTERS clusters, writing them to output. Note that ordinary reps may be
 *          followers of several clusters, but cluster heads may not.
 * @param io_nEstimatedFollowers is altered: followers of a cluster head have their individual follower count set to zero.
 */
__device__ void block_extract_clusters(const int nReps,
                                       const int nMetaPartitions,
                                       const int minClusterLen,
                                       const int secLen,
                                       scalar *in_reps,
                                       unsigned int *in_repMasks,
                                       unsigned int *in_ownFollowerMasks,
                                       scalar *in_current,
                                       unsigned int *io_nEstimatedFollowers,
                                       scalar *out_clusters,
                                       int *out_clusterLen,
                                       scalar *out_clusterCurrent)
{
    const unsigned laneid = threadIdx.x & 31;
    const unsigned warpid = threadIdx.x >> 5;
    __shared__ int shared_cache[64];

    for ( int clusterIdx = 0; clusterIdx < MAXCLUSTERS; clusterIdx++ ) {
        unsigned int repIdx, bestRepIdx = threadIdx.x;
        int nEstimatedFollowers = 0, mostEstimatedFollowers = 0;

        // Block-stride reduction
        for ( repIdx = threadIdx.x; repIdx < nReps; repIdx += blockDim.x ) {
            nEstimatedFollowers = io_nEstimatedFollowers[repIdx];
            if ( nEstimatedFollowers > mostEstimatedFollowers ) {
                mostEstimatedFollowers = nEstimatedFollowers;
                bestRepIdx = repIdx;
            }
        }

        // Reduce in all warps, save to cache
        for ( unsigned int offset = 1; offset < warpSize; offset <<= 1 ) {
            nEstimatedFollowers = __shfl_down_sync(0xffffffff, mostEstimatedFollowers, offset);
            repIdx = __shfl_down_sync(0xffffffff, bestRepIdx, offset);
            if ( nEstimatedFollowers > mostEstimatedFollowers ) {
                mostEstimatedFollowers = nEstimatedFollowers;
                bestRepIdx = repIdx;
            }
        }
        if ( laneid == 0 ) {
            shared_cache[2*warpid] = bestRepIdx;
            shared_cache[2*warpid + 1] = mostEstimatedFollowers;
        }

        __syncthreads();

        // Final reduce in first warp
        if ( warpid == 0 ) {
            if ( laneid < (blockDim.x>>5) ) {
                bestRepIdx = shared_cache[2*laneid];
                mostEstimatedFollowers = shared_cache[2*laneid + 1];
            } else {
                bestRepIdx = mostEstimatedFollowers = 0;
            }
            for ( unsigned int offset = 1; offset < (blockDim.x>>5); offset <<= 1 ) {
                nEstimatedFollowers = __shfl_down_sync(0xffffffff, mostEstimatedFollowers, offset);
                repIdx = __shfl_down_sync(0xffffffff, bestRepIdx, offset);
                if ( nEstimatedFollowers > mostEstimatedFollowers ) {
                    mostEstimatedFollowers = nEstimatedFollowers;
                    bestRepIdx = repIdx;
                }
            }

            // broadcast from lane 0 to the entire warp
            bestRepIdx = __shfl_sync(0xffffffff, bestRepIdx, 0);
            mostEstimatedFollowers = __shfl_sync(0xffffffff, mostEstimatedFollowers, 0);

            // If the largest remaining cluster is valid...
            if ( mostEstimatedFollowers * secLen >= minClusterLen ) {
                // Calculate cluster rep deviation vector norm (build_section_primitives normalises for deltabar and secLen only)
                scalar norm = 0;
                for ( int paramIdx = laneid; paramIdx < NPARAMS; paramIdx += warpSize ) {
                    scalar tmp = in_reps[bestRepIdx*NPARAMS + paramIdx];
                    norm += tmp*tmp;
                }
                norm = std::sqrt(warpReduceSum(norm));

                // Write normalised cluster rep to output
                for ( int paramIdx = laneid; paramIdx < NPARAMS; paramIdx += warpSize )
                    out_clusters[blockIdx.x*MAXCLUSTERS*NPARAMS + clusterIdx*NPARAMS + paramIdx] \
                            = in_reps[bestRepIdx*NPARAMS + paramIdx] / norm;

                if ( laneid == 0 ) {
                    out_clusterLen[blockIdx.x*MAXCLUSTERS + clusterIdx] = mostEstimatedFollowers * secLen;

                    // Write cluster rep idx to shared
                    shared_cache[0] = bestRepIdx;
                }

                // Collect current across cluster
                scalar current = 0;
                unsigned int laneBit = (1 << laneid);
                for ( int metaPartitionIdx = 0; metaPartitionIdx < nMetaPartitions; metaPartitionIdx++ ) {
                    unsigned int mask = in_repMasks[metaPartitionIdx * nReps + bestRepIdx];
                    repIdx = metaPartitionIdx << 5;
                    for ( unsigned int repBit = 1; repBit && repIdx<nReps; repBit<<=1, repIdx++ ) {
                        if ( mask & repBit ) {
                            unsigned int ownFollowerMask = in_ownFollowerMasks[repIdx];
                            if ( ownFollowerMask & laneBit )
                                current += in_current[blockIdx.x * (nReps/2) * PARTITION_SIZE + (repIdx/2) * PARTITION_SIZE + laneid];
                        }
                    }
                }
                current = warpReduceSum(current);
                if ( laneid == 0 )
                    out_clusterCurrent[blockIdx.x*MAXCLUSTERS + clusterIdx] = current / mostEstimatedFollowers;

            } else if ( laneid == 0 ) {
                // Largest remaining cluster is too short, so bail across the block
                shared_cache[0] = nReps;
                out_clusterLen[blockIdx.x*MAXCLUSTERS + clusterIdx] = 0;
            }
        }

        __syncthreads();

        // Check for completion
        bestRepIdx = shared_cache[0];
        if ( bestRepIdx == nReps )
            return;

        // Remove cluster members' ability to champion another cluster
        // This is much leaner than the original CPU algorithm, where the cluster was completely removed, necessitating rebuilding
        // nEstimatedFollowers from nOwnFollowers for all remaining reps
        for ( repIdx = threadIdx.x; repIdx < nReps; repIdx += blockDim.x ) {
            // Is repIdx similar to bestRepIdx <==> is bestRepIdx's bit set in repIdx's similarity mask?
            if ( in_repMasks[(bestRepIdx>>5) * nReps + repIdx] & (1<<(bestRepIdx&31)) ) {
                io_nEstimatedFollowers[repIdx] = 0;
            }
        }

        __syncthreads();
    }
}

// Launch: One 1-D block per stim
// dynamic shared mem size: nReps * NPARAMS * sizeof(scalar)       [for sh_reps]
//                        + nReps * 2 * sizeof(int)                [for rep follower counts, own (as mask) and estimated (as number)]
//                        + nReps * nMetaPartitions * sizeof(int)  [for similarity masks]
/**
 * @brief find_cluster_representatives builds clusters from partitioned section primitives. Each cluster is reported by its leading section,
 *          rather than a true mean, and heuristics are applied to estimate the cluster duration
 * @param in_contrib -- output from build_section_primitives
 */
__global__ void find_cluster_representatives(const int nSecs,
                                             const int nPartitions,
                                             const int nMetaPartitions,
                                             const scalar dotp_threshold,
                                             const int secLen,
                                             const int minClusterLen,
                                             scalar *in_contrib,
                                             scalar *in_current,
                                             scalar *out_clusters,
                                             int *out_clusterLen,
                                             scalar *out_clusterCurrent)
{
    const int nReps = 2*nPartitions;
    extern __shared__ unsigned int shared_mem[];
    unsigned int *sh_ownFollowerMasks = shared_mem;
    unsigned int *sh_nEstimatedFollowers = sh_ownFollowerMasks + nReps;
    unsigned int *sh_repMasks = sh_nEstimatedFollowers + nReps;
    scalar *sh_reps = (scalar*)&sh_repMasks[nReps*nMetaPartitions];

    // init shared mem                followers x2    masks             parameter contribs for each rep
    for ( int i = threadIdx.x, iEnd = 2*nReps + nReps*nMetaPartitions + (sizeof(scalar)/sizeof(int)) * nReps * NPARAMS;
          i < iEnd; i += blockDim.x )
        shared_mem[i] = 0;
    __syncthreads();

    // Compare sections within partitions, find the two most representative sections, and file them into sh_reps, sh_nOwnFollowers
    find_section_representatives(nSecs, nPartitions, dotp_threshold, in_contrib, sh_reps, sh_ownFollowerMasks);

    __syncthreads();

    // Compare representative sections across the entire stim, populating their nEstimatedFollowers and repMasks
    estimate_rep_follower_count(nReps, nMetaPartitions, dotp_threshold, sh_reps, sh_ownFollowerMasks, sh_nEstimatedFollowers, sh_repMasks);

    __syncthreads();

    // Extract the largest clusters one by one until hitting either the MAXCLUSTERS limit, or the minClusterLen one
    block_extract_clusters(nReps, nMetaPartitions, minClusterLen, secLen,
                           sh_reps, sh_repMasks, sh_ownFollowerMasks, in_current, sh_nEstimatedFollowers,
                           out_clusters, out_clusterLen, out_clusterCurrent);
}

extern "C" void pullClusters(int nStims)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nStims * MAXCLUSTERS * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterLen, d_clusterLen, nStims * MAXCLUSTERS * sizeof(int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterCurrent, d_clusterCurrent, nStims * MAXCLUSTERS * sizeof(scalar), cudaMemcpyDeviceToHost));
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
    int nReps = 2*nPartitions;
    int nMetaPartitions = (nReps + 31)/32; // count one for every 32 representatives

    resizeArrayPair(sections, d_sections, sections_size, nStims * nPartitions * NPARAMS * PARTITION_SIZE);
    resizeArray(d_currents, currents_size, nStims * nPartitions * PARTITION_SIZE);
    resizeArrayPair(clusters, d_clusters, clusters_size, nClusters * NPARAMS);
    resizeArrayPair(clusterLen, d_clusterLen, clusterLen_size, nClusters);
    resizeArrayPair(clusterCurrent, d_clusterCurrent, clusterCurrent_size, nClusters);

    scalar deltabar_array[NPARAMS];
    for ( int i = 0; i < NPARAMS; i++ )
        deltabar_array[i] = deltabar_arg[i];
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(deltabar, deltabar_array, NPARAMS*sizeof(scalar)));

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    build_section_primitives<<<grid, block>>>(trajLen, nTraj, nStims, duration, secLen, nPartitions, d_sections, d_currents);

    size_t shmem_size = nReps * NPARAMS * sizeof(scalar)
                      + nReps * 2 * sizeof(int)
                      + nReps * nMetaPartitions * sizeof(int);

    find_cluster_representatives<<<nStims, 512, shmem_size>>>(nSecs, nPartitions, nMetaPartitions, dotp_threshold, secLen, minClusterLen,
                                                              d_sections, d_currents,
                                                              d_clusters, d_clusterLen, d_clusterCurrent);

    if ( pull_results )
        pullClusters(nStims);
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
                            contrib += dd_timeseries[t*NMODELS + id0 + i + i/(trajLen-1) + 1];
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
