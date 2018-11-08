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
static scalar *d_gradient;
static constexpr unsigned int gradientSz = NPAIRS * (NPAIRS - 1); // No diagonal

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

#endif
