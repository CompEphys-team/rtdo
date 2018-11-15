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
///  >============================     Elementary Effects WG    ================================================================<
/// ******************************************************************************************************************************

static constexpr int STIMS_PER_CLUSTER_WARP = NPARAMS>16 ? 1 : NPARAMS>8 ? 2 : NPARAMS>4 ? 4 : NPARAMS>2 ? 8 : NPARAMS>1 ? 16 : 32;
static constexpr int STIMS_PER_CLUSTER_BLOCK = 8 * STIMS_PER_CLUSTER_WARP;

// Code adapated from Justin Luitjens, <https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/>
// Note, shuffle is only supported on compute capability 3.x and higher
__device__ inline scalar warpReduceSum(scalar val, int cutoff = warpSize)
{
    for ( int offset = 1; offset < cutoff; offset *= 2 )
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ inline scalar sumOverStim(scalar val, int stimWidth, int stimIdx)
{
    static __shared__ scalar tmp[STIMS_PER_CLUSTER_BLOCK];
    val = warpReduceSum(val, stimWidth);
    if ( NPARAMS > warpSize ) {
        if ( threadIdx.x & (stimWidth-1) == 0 ) // Lane 0
            atomicAdd(&tmp[stimIdx], val);
        __syncthreads();
        val = tmp[stimIdx];
    } else {
        val = __shfl_sync(0xffffffff, val, 0, stimWidth);
    }
    return val;
}

__global__ void clusterKernel(int trajLen,
                              int nTraj,
                              int nStims,
                              int duration,
                              int secLen,
                              scalar dotp_threshold,
                              scalar *global_clusters, int *global_clusterLen)
{
    static constexpr int stimWidth = 32/STIMS_PER_CLUSTER_WARP;
    const int nUsefulTraces = (trajLen-1)*nTraj;
    int paramIdx, stimIdx;
    if ( STIMS_PER_CLUSTER_WARP == 1 ) {
        // One row, one stim
        paramIdx = threadIdx.x;
        stimIdx = threadIdx.y;
    } else {
        // Each row is one warp with multiple stims
        paramIdx = threadIdx.x % stimWidth;
        stimIdx = (threadIdx.y * STIMS_PER_CLUSTER_WARP) + (threadIdx.x/stimWidth);
    }
    const int global_stimIdx = STIMS_PER_CLUSTER_BLOCK*blockIdx.x + stimIdx;
    const int id0 = global_stimIdx * trajLen * nTraj; // id of the first trajectory's starting point model
    const int nContrib = (nUsefulTraces/NPARAMS) + (paramIdx < (nUsefulTraces%NPARAMS));
    const iObservations obs = dd_obsUNI[id0];
    int nextObs = 0;

    __shared__ scalar sh_clusters[STIMS_PER_CLUSTER_BLOCK][MAXCLUSTERS][NPARAMS];
    __shared__ scalar sh_cluster_square_norm[STIMS_PER_CLUSTER_BLOCK][MAXCLUSTERS];
    __shared__ int sh_clusterLen[STIMS_PER_CLUSTER_BLOCK][MAXCLUSTERS];
    { // zero-initialise
        if ( paramIdx < NPARAMS )
            for ( int i = 0; i < MAXCLUSTERS; i++ )
                sh_clusters[stimIdx][i][paramIdx] = 0;

        int laneid = threadIdx.x + blockDim.x*threadIdx.y;
        if ( laneid < STIMS_PER_CLUSTER_BLOCK ) {
            for ( int i = 0; i < MAXCLUSTERS; i++ ) {
                sh_cluster_square_norm[laneid][i] = 0;
                sh_clusterLen[laneid][i] = 0;
            }
        }
        __syncthreads();
    }

    // Construct clusters in shared memory
    scalar contrib, square;
    int t = 0;
    while ( t < duration ) {

        // Load ee contribution from this parameter
        // Note, timeseries[0,32,64,...] are unused (COMPARE_PREVTHREAD), hence "+ i/31 + 1" indexing
        contrib = 0;
        int trueSecLen = 0;
        for ( int tEnd = t + secLen; t < tEnd; t++ ) {
            if ( paramIdx < NPARAMS
                 && global_stimIdx < nStims
                 && nextObs < iObservations::maxObs
                 && t >= obs.start[nextObs] ) {
                if ( t < obs.stop[nextObs] ) {
                    for ( int i = paramIdx; i < nUsefulTraces; i += NPARAMS )
                        contrib += dd_timeseries[t*NMODELS + id0 + i + i/(trajLen-1) + 1];
                    ++trueSecLen;
                } else {
                    ++nextObs;
                }
            }
        }

        // Normalise contrib to the mean deviation per tick, per single detune
        if ( trueSecLen )
            contrib /= trueSecLen * nContrib * deltabar[paramIdx];
        // Compute the square norm (sum of squared contributions) for scalar product normalisation
        square = contrib * contrib;
        square = sumOverStim(square, stimWidth, stimIdx);

        // Compute the scalar product with each existing cluster to find the closest match
        scalar max_dotp = 0;
        int closest_cluster = 0;
        int nClusters = 0;
        for ( int i = 0; i < MAXCLUSTERS; ++i ) {
            scalar dotp = 0;

            // Ignore empty sections (all zeroes is (a) useless and (b) likely in an unobserved region)
            if ( square > 0 ) {
                if ( sh_clusterLen[stimIdx][i] > 0 ) {
                    ++nClusters;
                    if ( paramIdx < NPARAMS )
                        dotp = contrib * sh_clusters[stimIdx][i][paramIdx];
                }
                dotp = sumOverStim(dotp, stimWidth, stimIdx);
                dotp /= std::sqrt(square * sh_cluster_square_norm[stimIdx][i]); // normalised
                if ( (max_dotp >= 0 && dotp > max_dotp) || (max_dotp < 0 && dotp < max_dotp) ) {
                    max_dotp = dotp;
                    closest_cluster = i;
                }
            } else if ( NPARAMS > warpSize ) {
                // sumOverStim has a __syncthreads call, idle over it to prevent stalling
                dotp = sumOverStim(dotp, stimWidth, stimIdx);
            }
        }

        // No adequate cluster: start a new one
        if ( scalarfabs(max_dotp) < dotp_threshold ) {
            closest_cluster = nClusters;
        }

        // Add present contribution to the nearest cluster, ignoring empty sections and cluster overflow
        if ( square > 0 && closest_cluster < MAXCLUSTERS ) {
            if ( paramIdx < NPARAMS ) {
                contrib += sh_clusters[stimIdx][closest_cluster][paramIdx] * (max_dotp < 0 ? -1 : 1);
                sh_clusters[stimIdx][closest_cluster][paramIdx] = contrib;
            }
            square = contrib * contrib;
            square = sumOverStim(square, stimWidth, stimIdx);
            if ( paramIdx == 0 ) {
                sh_cluster_square_norm[stimIdx][closest_cluster] = square;
                ++sh_clusterLen[stimIdx][closest_cluster];
            }
        }
        __syncthreads();
    }

    // Push normalised completed clusters to global memory
    if ( paramIdx < NPARAMS ) {
        for ( int i = 0; i < MAXCLUSTERS; i++ ) {
            contrib = sh_clusters[stimIdx][i][paramIdx];
            square = contrib*contrib;
            global_clusters[(((global_stimIdx * MAXCLUSTERS) + i) * NPARAMS) + paramIdx] =
                    square / sh_cluster_square_norm[stimIdx][i];
            if ( paramIdx == 0 ) {
                global_clusterLen[(global_stimIdx * MAXCLUSTERS) + i] = sh_clusterLen[stimIdx][i];
            }
        }
    }
}

extern "C" void cluster(int trajLen, /* length of EE trajectory (power of 2, <=32) */
                        int nTraj, /* Number of EE trajectories */
                        int duration,
                        int secLen,
                        scalar dotp_threshold,
                        std::vector<scalar> deltabar_arg)
{
    unsigned int nStims = NMODELS / (trajLen*nTraj);
    unsigned int nClusters = nStims * MAXCLUSTERS;
    resizeArrayPair(clusters, d_clusters, clusters_size, nClusters * NPARAMS);
    resizeArrayPair(clusterLen, d_clusterLen, clusterLen_size, nClusters);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(deltabar, deltabar_arg.data(), NPARAMS*sizeof(scalar)));

    dim3 block(((NPARAMS+31)/32)*32, STIMS_PER_CLUSTER_BLOCK/STIMS_PER_CLUSTER_WARP);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));
    clusterKernel<<<grid, block>>>(trajLen, nTraj, nStims, duration, secLen, dotp_threshold, d_clusters, d_clusterLen);

    CHECK_CUDA_ERRORS(cudaMemcpy(clusters, d_clusters, nClusters * NPARAMS * sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(clusterLen, d_clusterLen, nClusters * sizeof(int), cudaMemcpyDeviceToHost));
}



__global__ void find_deltabar_kernel(int trajLen, int nTraj, int nStims, int duration, scalar *global_clusters, int *global_clusterLen)
{
    static constexpr int stimWidth = 32/STIMS_PER_CLUSTER_WARP;
    const int nUsefulTraces = (trajLen-1)*nTraj;
    int paramIdx, stimIdx;
    if ( STIMS_PER_CLUSTER_WARP == 1 ) {
        // One row, one stim
        paramIdx = threadIdx.x;
        stimIdx = threadIdx.y;
    } else {
        // Each row is one warp with multiple stims
        paramIdx = threadIdx.x % stimWidth;
        stimIdx = (threadIdx.y * STIMS_PER_CLUSTER_WARP) + (threadIdx.x/stimWidth);
    }
    const int global_stimIdx = STIMS_PER_CLUSTER_BLOCK*blockIdx.x + stimIdx;
    const int id0 = global_stimIdx * trajLen * nTraj;
    const int nContrib = (nUsefulTraces/NPARAMS) + (paramIdx < (nUsefulTraces%NPARAMS));
    const iObservations obs = dd_obsUNI[id0];
    int nextObs = 0;
    int nSamples = 0;

    __shared__ scalar sh_clusters[STIMS_PER_CLUSTER_BLOCK][NPARAMS];
    __shared__ scalar sh_nSamples[STIMS_PER_CLUSTER_BLOCK];

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
    const int tid = threadIdx.y*blockDim.x + threadIdx.x;
    for ( int width = STIMS_PER_CLUSTER_BLOCK; width > 0; width /= 32 ) {
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

extern "C" std::vector<scalar> find_deltabar(int trajLen, int nTraj, int duration)
{
    unsigned int nStims = NMODELS / (trajLen*nTraj);
    dim3 block(((NPARAMS+31)/32)*32, STIMS_PER_CLUSTER_BLOCK/STIMS_PER_CLUSTER_WARP);
    dim3 grid(((nStims+STIMS_PER_CLUSTER_BLOCK-1)/STIMS_PER_CLUSTER_BLOCK));

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
