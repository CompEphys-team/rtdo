#ifndef LIB_DEFINITIONS_H
#define LIB_DEFINITIONS_H

#include "universallibrary.h"
#include "cuda_helper.h"

#include <curand.h>
#include <cusolverDn.h>

scalar *target = nullptr, *d_target = nullptr;
__constant__ scalar *dd_target = nullptr;
unsigned int target_size = 0, latest_target_size = 0;

scalar *timeseries = nullptr, *d_timeseries = nullptr;
__constant__ scalar *dd_timeseries = nullptr;
unsigned int timeseries_size = 0, latest_timeseries_size = 0;

scalar *summary = nullptr, *d_summary = nullptr;
__constant__ scalar *dd_summary = nullptr;
unsigned int summary_size = 0, latest_summary_size = 0;

__constant__ iStimulation singular_stim;
__constant__ iObservations singular_obs;

__constant__ scalar singular_clampGain;
__constant__ scalar singular_accessResistance;
__constant__ int singular_iSettleDuration;
__constant__ scalar singular_Imax;
__constant__ scalar singular_dt;

__constant__ size_t singular_targetOffset;

// profiler memory space
scalar *d_prof_error, *d_prof_dist_uw, *d_prof_dist_to, *d_prof_dist_w;
constexpr unsigned int profSz = NMODELS * (NMODELS - 1) / 2; // No diagonal

scalar *d_random = nullptr;
__constant__ scalar *dd_random = nullptr;
unsigned int random_size = 0;
curandGenerator_t cuRNG;

// wavegen memory space
scalar *clusters = nullptr, *d_clusters = nullptr;
unsigned int clusters_size = 0;

int *clusterLen = nullptr, *d_clusterLen = nullptr;
unsigned int clusterLen_size = 0;

unsigned int *d_clusterMasks = nullptr;
unsigned int clusterMasks_size = 0;

scalar *clusterCurrent = nullptr, *d_clusterCurrent = nullptr;
unsigned int clusterCurrent_size = 0;

scalar *sections = nullptr, *d_sections = nullptr;
unsigned int sections_size = 0;

scalar *d_currents= nullptr;
unsigned int currents_size = 0;

iObservations *clusterObs = nullptr, *d_clusterObs = nullptr;
unsigned int clusterObs_size = 0;

Bubble *bubbles = nullptr, *d_bubbles = nullptr;
unsigned int bubbles_size = 0;

__constant__ scalar deltabar[NPARAMS];

__constant__ unsigned char detuneParamIndices[NMODELS];
__constant__ unsigned short numDetunesByParam[NPARAMS];

constexpr int PARTITION_SIZE = 32;
constexpr int STIMS_PER_CLUSTER_BLOCK =
        ( (16*NPARAMS*(PARTITION_SIZE+1) + 16*(PARTITION_SIZE+1)) * sizeof(scalar) < 0xc000 )
        ? 16
        : ( (8*NPARAMS*(PARTITION_SIZE+1) + 8*(PARTITION_SIZE+1)) * sizeof(scalar) < 0xc000 )
        ? 8
        : ( (4*NPARAMS*(PARTITION_SIZE+1) + 4*(PARTITION_SIZE+1)) * sizeof(scalar) < 0xc000 )
        ? 4
        : ( (2*NPARAMS*(PARTITION_SIZE+1) + 2*(PARTITION_SIZE+1)) * sizeof(scalar) < 0xc000 )
        ? 2
        : 1;


// PCA (cuSolver) space and function macros
cusolverDnHandle_t cusolverH = NULL;
scalar *PCA_d_S = nullptr, *PCA_d_U = nullptr, *PCA_d_VT = nullptr, *PCA_d_lwork = nullptr;
unsigned int PCA_S_size = 0, PCA_U_size = 0, PCA_VT_size = 0, PCA_lwork_size = 0;
scalar *PCA_TL = nullptr;
unsigned int PCA_TL_size = 0;

#ifdef USEDOUBLE
#define cusolverDn__scalar__gesvd_bufferSize cusolverDnDgesvd_bufferSize
#define cusolverDn__scalar__gesvd cusolverDnDgesvd
#else
#define cusolverDn__scalar__gesvd_bufferSize cusolverDnSgesvd_bufferSize
#define cusolverDn__scalar__gesvd cusolverDnSgesvd
#endif


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


void pushDeltabar(std::vector<double> dbar);
std::vector<unsigned short> pushDetuneIndices(int trajLen, int nTraj, const MetaModel &model);

__global__ void build_section_primitives(const int trajLen,
                                         const int nTraj,
                                         const int nStims,
                                         const int duration,
                                         const int secLen,
                                         const int nPartitions,
                                         const bool VC,
                                         scalar *out_sections,
                                         scalar *out_current);

#endif // LIB_DEFINITIONS_H
