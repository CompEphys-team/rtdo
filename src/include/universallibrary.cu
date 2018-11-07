#ifndef UNIVERSAL_CU
#define UNIVERSAL_CU

#include "universallibrary.h"
#include "cuda_helper.h"

#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

static scalar *target = nullptr, *d_target = nullptr;
static __device__ scalar *dd_target = nullptr;
static unsigned int target_size = 0, latest_target_size = 0;

static scalar *timeseries = nullptr, *d_timeseries = nullptr;
static __device__ scalar *dd_timeseries = nullptr;
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
static scalar *d_gradient, *d_err;
static constexpr unsigned int gradientSz = NPAIRS * (NPAIRS - 1); // No diagonal
static constexpr unsigned int errSz = gradientSz/2; // Above diagonal only

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
    CHECK_CUDA_ERRORS(cudaMalloc(&d_err, errSz * sizeof(scalar)));
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
    unsigned int tmp = target_size;
    resizeHostArray(target, tmp, newSize);
    resizeArray(d_target, target_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_target, &d_target, sizeof(scalar*)));
    latest_target_size = newSize;
}

extern "C" void pushTarget()
{
    CHECK_CUDA_ERRORS(cudaMemcpy(d_target, target, latest_target_size * sizeof(scalar), cudaMemcpyHostToDevice))
}

extern "C" void resizeOutput(size_t newSize)
{
    unsigned int tmp = timeseries_size;
    resizeHostArray(timeseries, tmp, newSize);
    resizeArray(d_timeseries, timeseries_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_timeseries, &d_timeseries, sizeof(scalar*)));
    latest_timeseries_size = newSize;
}

extern "C" void pullOutput()
{
    CHECK_CUDA_ERRORS(cudaMemcpy(timeseries, d_timeseries, latest_timeseries_size * sizeof(scalar), cudaMemcpyDeviceToHost))
}



/// Profiler kernel & host function
// Compute the current deviation of both tuned and untuned models against each tuned model
// Models are interleaved (even id = tuned, odd id = detuned) in SamplingProfiler
__global__ void compute_err_and_gradient(unsigned int nSamples, int stride, scalar *targetParam, scalar *gradient, scalar *err)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // probe
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; // reference
    scalar tuned_err = 0, detuned_err = 0;
    for ( unsigned int i = 0; i < nSamples; i += stride ) {
        tuned_err += scalarfabs(dd_timeseries[2*x + NMODELS*i] - dd_timeseries[2*y + NMODELS*i]);
        detuned_err += scalarfabs(dd_timeseries[2*x+1 + NMODELS*i] - dd_timeseries[2*y + NMODELS*i]);
    }

    if ( x != y ) // Ignore diagonal (don't probe against self)
        //       error differential relative to detuning direction   invert sign as appropriate; positive differential indicates gradient pointing towards reference
        gradient[x + NPAIRS*y - (y+1)] = (detuned_err - tuned_err) * (1 - 2 * (targetParam[2*x] < targetParam[2*y]));

    if ( x > y ) // (tuned) err is symmetrical; keep only values above the diagonal for faster median finding
        err[x + NPAIRS*y - (y+1)*(y+2)/2] = tuned_err;
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
    dim3 grid(NPAIRS/32, NPAIRS/16);
    compute_err_and_gradient<<<grid, block>>>(nSamples, stride, d_targetParam, d_gradient, d_err);

    thrust::device_ptr<scalar> gradient = thrust::device_pointer_cast(d_gradient);
    thrust::device_ptr<scalar> err = thrust::device_pointer_cast(d_err);
    thrust::sort(gradient, gradient + gradientSz);
    thrust::sort(err, err + errSz);

    double nPositive = thrust::count_if(gradient, gradient + gradientSz, is_positive());
    accuracy = nPositive / gradientSz;

    scalar median_g[2], median_e[2];
    CHECK_CUDA_ERRORS(cudaMemcpy(median_g, d_gradient + gradientSz/2, 2*sizeof(scalar), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(median_e, d_err + errSz/2, 2*sizeof(scalar), cudaMemcpyDeviceToHost));
    scalar median_gradient = (median_g[0] + median_g[1]) / 2;
    scalar median_err = (median_e[0] + median_e[1]) / 2;
    median_norm_gradient = median_gradient / median_err;
}

#endif
