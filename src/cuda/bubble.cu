#include "lib_definitions.h"

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
                             const bool VC,
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
    if ( warpid == 0 ) {
        sh_current[laneid] = 0;
        sh_deviation[laneid].zero();
    }
    __syncthreads();

    // Gather bubble deviation and current
    for ( unsigned int secIdx = start.startCycle, lastSec = start.startCycle + start.cycles; secIdx < ((lastSec+31)&0xffffffe0); secIdx += blockDim.x ) {
        scalar current;
        if ( secIdx < lastSec ) {
            dev.load(in_contrib + stimIdx * NPARAMS * nSecs + secIdx, nSecs);
            if ( VC )
                current = in_current[stimIdx * nSecs + secIdx];
        } else {
            dev.zero();
            current = 0;
        }

        if ( VC )
            current = warpReduceSum(current);
        dev = warpReduceSum(dev);

        if ( laneid == 0 ) {
            if ( VC )
                sh_current[warpid] += current;
            sh_deviation[warpid] += dev;
        }
    }

    __syncthreads();

    if ( warpid == 0 ) {
        scalar current;
        if ( VC )
            current = warpReduceSum(sh_current[laneid]);
        dev = warpReduceSum(sh_deviation[laneid]);
        if ( laneid == 0 ) {
            if ( VC )
                out_bubbleCurrents[stimIdx * NPARAMS + targetParamIdx] = current/start.cycles;
            dev /= std::sqrt(dev.dotp(dev));
            dev.store(out_deviations + stimIdx * NPARAMS * NPARAMS + targetParamIdx * NPARAMS);
        }
    }
}

extern "C" void pullBubbles(int nStims, bool includeCurrent)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(bubbles, d_bubbles, nStims * NPARAMS * sizeof(Bubble), cudaMemcpyDeviceToHost));
    if ( includeCurrent )
        CHECK_CUDA_ERRORS(cudaMemcpy(clusterCurrent, d_clusterCurrent, nStims * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nStims * NPARAMS * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
}

extern "C" void bubble(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                       int nTraj, /* Number of EE trajectories */
                       int duration,
                       int secLen,
                       std::vector<double> deltabar_arg,
                       const MetaModel &model,
                       bool VC,
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

    pushDeltabar(deltabar_arg);
    pushDetuneIndices(trajLen, nTraj, model);

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    build_section_primitives<<<grid, block>>>(trajLen, nTraj, nStims, duration, secLen, nPartitions, VC, d_sections, d_currents);

    size_t nWarps = 16;
    size_t shmem_for_bubbles = 3 * nPartitions * sizeof(Bubble);
    size_t shmem_for_stats = 32 * sizeof(scalar) + 32 * sizeof(Parameters);
    size_t shmem_size = std::max(shmem_for_bubbles, shmem_for_stats);
    block = dim3(nWarps * 32);
    grid = dim3(nStims, NPARAMS);
    buildBubbles<<<grid, block, shmem_size>>>(nPartitions, secLen, VC, d_sections, d_currents, d_clusters, d_clusterCurrent, d_bubbles);

    if ( pull_results )
        pullBubbles(nStims, VC);
}
