#include "lib_definitions.h"

/**
 * @brief build_section_primitives chops the EE traces in d_timeseries into deltabar-normalised deviation vectors ("sections")
 *          representing up to secLen ticks. Sections are chunked into partitions of PARTITION_SIZE=32 sections each.
 *          Deviation vectors represent the mean deviation per tick, normalised to deltabar, caused by a single detuning.
 *          Note, this kernel expects the EE traces to be generated using TIMESERIES_COMPARE_NONE
 * @param VC Voltage clamp flag; if false, out_current is not touched, and the timeseries is assumed to be the pattern clamp pin current.
 * @param out_sections is the output, laid out as [stimIdx][paramIdx][partitionIdx][secIdx (local to partition)].
 * @param out_current is the mean current within each section, laid out as [stimIdx][partitionIdx][secIdx].
 */
__global__ void build_section_primitives(const int trajLen,
                                         const int nTraj,
                                         const int nStims,
                                         const int duration,
                                         const int secLen,
                                         const int nPartitions,
                                         const bool VC,
                                         scalar *out_sections,
                                         scalar *out_current)
{
    const int warpid = threadIdx.x / 32; // acts as block-local stim idx
    const int laneid = threadIdx.x & 31;
    const int stimIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // global stim idx; one stim per warp
    const int nTraces = trajLen*nTraj; // Total number of traces per stim, including starting point models
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
                    for ( int i = laneid; i < nLoads; i += warpSize ) { // Loop (warp-consistently) nLoads times
                        scalar current_mylane = 0;
                        if ( i < nTraces )
                            current_mylane = dd_timeseries[t*NMODELS + lane0_offset + i];
                        if ( VC ) {
                            scalar current_prevlane = __shfl_up_sync(0xffffffff, current_mylane, 1);
                            scalar diff = scalarfabs(current_prevlane - current_mylane);
                            if ( i < nTraces ) {
                                if ( i % trajLen != 0 ) // Add diff to detuned param's contribution (base points mute; atomic for NPARAMS<31)
                                    atomicAdd((scalar*)&sh_contrib[warpid][detuneParamIndices[i]][secIdx&31], diff);
                                current_mylane = scalarfabs(current_mylane);
                            }
                            current_mylane = warpReduceSum(current_mylane);
                            if ( laneid == 0 )
                                sh_current[warpid][secIdx&31] += current_mylane;
                        } else {
                            if ( i < nTraces && i % trajLen != 0 )
                                atomicAdd((scalar*)&sh_contrib[warpid][detuneParamIndices[i]][secIdx&31], scalarfabs(current_mylane));
                        }
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
            if ( t < duration || laneid <= (secIdx&31) ) {
                for ( int paramIdx = 0; paramIdx < NPARAMS; paramIdx++ ) {
                    out_sections[stimIdx * NPARAMS * nPartitions * PARTITION_SIZE
                            + paramIdx * nPartitions * PARTITION_SIZE
                            + partitionIdx * PARTITION_SIZE
                            + laneid]
                            = trueSecLen_static
                              ? sh_contrib[warpid][paramIdx][laneid] / (trueSecLen_static * deltabar[paramIdx] * numDetunesByParam[paramIdx])
                              : 0;
                    sh_contrib[warpid][paramIdx][laneid] = 0;
                }
            }

            if ( VC ) {
                out_current[stimIdx * nPartitions * PARTITION_SIZE
                        + partitionIdx * PARTITION_SIZE
                        + laneid]
                        = sh_current[warpid][laneid] / (trueSecLen_static * nTraces);
                sh_current[warpid][laneid] = 0;
            }
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
 * @brief compare_between_partitions compares each section in one partition to all sections of another partition, setting mask bits in both
 * @param reference is the reference partition's deviation vector, warp-aligned (section i in thread i)
 * @param target is the target partition's deviation vector, warp-aligned
 * @param ref_norm is the reference vector's precomputed norm
 * @param dotp_threshold
 * @param target_mask is a reference to the target partition's mask and will be populated with the appropriate set bits
 * @return the reference partition's bitmask flagging each above-threshold similar section in target
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
                                const bool VC,
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
                    if ( VC ) {
                        current += in_current[blockIdx.x * nSecs + secIdx];
                        ++nAdditions;
                    }
                    tmp.load(in_contrib + blockIdx.x * NPARAMS * nSecs + secIdx, nSecs);
                    contrib += tmp;
                }
            }
            __syncwarp();

            // Reduce into lane 0
            for ( unsigned int i = 1; i < warpSize; i *= 2 ) {
                if ( VC ) {
                    current += __shfl_down_sync(0xffffffff, current, i);
                    nAdditions += __shfl_down_sync(0xffffffff, nAdditions, i);
                }
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

                if ( VC ) {
                    current /= nAdditions;
                    out_clusterCurrent[blockIdx.x * MAXCLUSTERS + clusterIdx] = current;
                }
            }
        }

        // Backstop
        if ( nClusters < MAXCLUSTERS && threadIdx.x == 0 ) {
            out_observations[blockIdx.x * MAXCLUSTERS + nClusters] = iObservations {{},{}};
        }
    }
}

extern "C" void pullClusters(int nStims, bool includeCurrent)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nStims * MAXCLUSTERS * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    if ( includeCurrent )
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
                       const MetaModel &model,
                       bool VC,
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

    pushDeltabar(deltabar_arg);
    pushDetuneIndices(trajLen, nTraj, model);

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    build_section_primitives<<<grid, block>>>(trajLen, nTraj, nStims, duration, secLen, nPartitions, VC, d_sections, d_currents);

    size_t shmem_size = std::max(32*nPartitions, 8192);
    size_t nWarps = 16;
    exactClustering<<<nStims, 32*nWarps, shmem_size*sizeof(int)>>>(nPartitions, dotp_threshold, secLen, minClusterLen, VC,
                                                                   d_sections, d_currents,
                                                                   d_clusters, d_clusterCurrent, d_clusterObs,
                                                                   d_clusterMasks, shmem_size);


    if ( pull_results )
        pullClusters(nStims, VC);
}
