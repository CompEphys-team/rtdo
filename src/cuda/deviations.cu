#include "lib_definitions.h"

/**
 * @brief get_posthoc_deviations is a reduced version of build_section_primitives designed to get the deviations vector and current for
 * stimulations with predefined observation windows.
 * @param out_clusters: Deviation vectors, [stimIdx][paramIdx]
 * @param out_current: Mean current, [stimIdx]
 */
__global__ void get_posthoc_deviations_kernel(const int trajLen,
                                              const int nTraj,
                                              const int nStims,
                                              const bool VC,
                                              scalar *out_clusters,
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

    volatile __shared__ scalar sh_contrib[STIMS_PER_CLUSTER_BLOCK][NPARAMS];
    for ( int i = threadIdx.x; i < STIMS_PER_CLUSTER_BLOCK*NPARAMS; i += blockDim.x )
        *((scalar*)sh_contrib + i) = 0;
    __syncthreads();

    if ( stimIdx >= nStims )
        return;

    // Accumulate current/contribs across observation period
    scalar current = 0;
    for ( int t = obs.start[0]; ; ++t ) {
        if ( t == obs.stop[nextObs] ) {
            if ( ++nextObs == iObservations::maxObs )
                break;
            t = obs.start[nextObs];
            if ( t == 0 )
                break;
        }

        __syncwarp();
        for ( int i = laneid; i < nLoads; i += warpSize ) {
            scalar current_mylane = 0;
            if ( i < nTraces )
                current_mylane = dd_timeseries[t*NMODELS + lane0_offset + i];
            if ( VC ) {
                scalar current_prevlane = __shfl_up_sync(0xffffffff, current_mylane, 1);
                scalar diff = scalarfabs(current_prevlane - current_mylane);
                if ( i < nTraces ) {
                    if ( i % trajLen != 0 )
                        atomicAdd((scalar*)&sh_contrib[warpid][detuneParamIndices[i]], diff);
                    current_mylane = scalarfabs(current_mylane);
                }
                current += warpReduceSum(current_mylane);
            } else if ( i < nTraces && i % trajLen != 0 ) {
                atomicAdd((scalar*)&sh_contrib[warpid][detuneParamIndices[i]], scalarfabs(current_mylane));
            }
        }
    }

    // Normalise
    __syncwarp();
    scalar dotp = 0;
    for ( int paramIdx = laneid; paramIdx < NPARAMS; paramIdx += warpSize ) {
        scalar contrib = sh_contrib[warpid][paramIdx] / (numDetunesByParam[paramIdx] * deltabar[paramIdx]);
        dotp += contrib * contrib;
        sh_contrib[warpid][paramIdx] = contrib;
    }

    __syncwarp();
    dotp = std::sqrt(warpReduceSum(dotp));
    for ( int paramIdx = laneid; paramIdx < NPARAMS; paramIdx += warpSize )
        out_clusters[stimIdx*NPARAMS + paramIdx] = sh_contrib[warpid][paramIdx] / dotp;

    if ( VC && laneid == 0 )
        out_current[stimIdx] = current / (obs.duration() * nTraces);
}

extern "C" void get_posthoc_deviations(int trajLen,
                                       int nTraj,
                                       unsigned int nStims,
                                       std::vector<double> deltabar_arg,
                                       const MetaModel &model,
                                       bool VC)
{
    resizeArrayPair(clusters, d_clusters, clusters_size, nStims * NPARAMS);
    resizeArrayPair(clusterCurrent, d_clusterCurrent, clusterCurrent_size, nStims);

    pushDeltabar(deltabar_arg);
    pushDetuneIndices(trajLen, nTraj, model);

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    get_posthoc_deviations_kernel<<<grid, block>>>(trajLen, nTraj, nStims, VC, d_clusters, d_clusterCurrent);

    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nStims * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    if ( VC )
        CHECK_CUDA_ERRORS(cudaMemcpy(clusterCurrent, d_clusterCurrent, nStims * sizeof(scalar), cudaMemcpyDeviceToHost));
}
