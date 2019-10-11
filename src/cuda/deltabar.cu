/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "lib_definitions.h"

__global__ void find_deltabar_kernel(const int trajLen,
                                     const int nTraj,
                                     const int nStims,
                                     scalar *out_sumSquares,
                                     int *out_nSamples)
{
    const int warpid = threadIdx.x / 32; // acts as block-local stim idx
    const int laneid = threadIdx.x & 31;
    const int stimIdx = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // global stim idx; one stim per warp
    const int nTraces = trajLen*nTraj; // Total number of traces per stim, including starting point models
    const int lane0_offset = stimIdx * nTraces;
    const int nLoads = (nTraces + 31) & 0xffffffe0;
    const iObservations obs = dd_obsUNI[stimIdx*nTraces];
    int nextObs = 0;

    volatile __shared__ scalar sh_sumSquares[STIMS_PER_CLUSTER_BLOCK][NPARAMS];
    for ( int i = threadIdx.x; i < STIMS_PER_CLUSTER_BLOCK*NPARAMS; i += blockDim.x )
        *((scalar*)sh_sumSquares + i) = 0;
    __syncthreads();

    if ( stimIdx >= nStims )
        return;

    // Accumulate square residuals
    int nSamples = 0;
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
            if ( (i < nTraces) && ((i % trajLen) != 0) ) {
                scalar diff = dd_timeseries[t*NMODELS + lane0_offset + i];
                atomicAdd((scalar*)&sh_sumSquares[warpid][detuneParamIndices[i]], diff*diff);
            }
        }
        ++nSamples;
    }

    // Combine sum squares across block
    __syncthreads();
    for ( int paramIdx = warpid; paramIdx < NPARAMS; paramIdx += STIMS_PER_CLUSTER_BLOCK ) {
        scalar sumSquares = 0;
        if ( laneid < STIMS_PER_CLUSTER_BLOCK )
            sumSquares = sh_sumSquares[laneid][paramIdx];
        sumSquares = warpReduceSum(sumSquares, STIMS_PER_CLUSTER_BLOCK);
        if ( laneid == 0 )
            out_sumSquares[blockIdx.x*NPARAMS + paramIdx] = sumSquares;
    }

    // Combine nSamples across block
    __syncthreads();
    if ( laneid == 0 )
        ((scalar*)sh_sumSquares)[warpid] = nSamples;
    __syncthreads();
    if ( warpid == 0 ) {
        nSamples = 0;
        if ( laneid < STIMS_PER_CLUSTER_BLOCK )
            nSamples = ((scalar*)sh_sumSquares)[laneid];
        __syncwarp();
        nSamples = warpReduceSum(nSamples, STIMS_PER_CLUSTER_BLOCK);
        if ( laneid == 0 )
            out_nSamples[blockIdx.x] = nSamples;
    }
}

extern "C" std::vector<double> find_deltabar(int trajLen, int nTraj, const MetaModel &model)
{
    unsigned int nStims = NMODELS / (trajLen*nTraj);

    dim3 block(STIMS_PER_CLUSTER_BLOCK * 32);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));

    resizeArrayPair(clusters, d_clusters, clusters_size, grid.x * NPARAMS);
    resizeArrayPair(clusterLen, d_clusterLen, clusterLen_size, grid.x);

    std::vector<unsigned short> nDetunes = pushDetuneIndices(trajLen, nTraj, model);

    find_deltabar_kernel<<<grid, block>>>(trajLen, nTraj, nStims, d_clusters, d_clusterLen);

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
        ret[p] = sqrt(ret[p] / (n * nDetunes[p]));
    return ret;
}
