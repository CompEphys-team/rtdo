#ifndef PROFILER_CU
#define PROFILER_CU

#include "profilerlibrary.h"
#include "cuda_helper.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

static __device__ scalar *dd_current;
static __device__ scalar *dd_gradient;
static __device__ scalar *dd_err;
static scalar *d_current, *d_gradient, *d_err;
static unsigned int currentSz = 1024;
static constexpr unsigned int gradientSz = NCOMP - NPAIRS; // No diagonal
static constexpr unsigned int errSz = (NCOMP - NPAIRS)/2; // Above diagonal only

static __constant__ Stimulation *dd_stim;

void libInit(ProfilerLibrary::Pointers &pointers)
{
    CHECK_CUDA_ERRORS(cudaHostAlloc(&pointers.stim, sizeof(Stimulation), cudaHostAllocPortable));
        deviceMemAllocate(&pointers.d_stim, dd_stim, sizeof(Stimulation));

    deviceMemAllocate(&d_current, dd_current, currentSz * NMODELS * sizeof(scalar));
    deviceMemAllocate(&d_gradient, dd_gradient, gradientSz * sizeof(scalar));
    deviceMemAllocate(&d_err, dd_err, errSz * sizeof(scalar));

    allocateMem();
    initialize();
}

extern "C" void libExit(ProfilerLibrary::Pointers &pointers)
{
    freeMem();
    CHECK_CUDA_ERRORS(cudaFreeHost(pointers.stim));
    CHECK_CUDA_ERRORS(cudaFree(pointers.d_stim));
    CHECK_CUDA_ERRORS(cudaFree(d_current));
    CHECK_CUDA_ERRORS(cudaFree(d_gradient));
    CHECK_CUDA_ERRORS(cudaFree(d_err));
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

void resize_current(unsigned int nSamples)
{
    if ( currentSz < nSamples ) {
        CHECK_CUDA_ERRORS(cudaFree(d_current));
        deviceMemAllocate(&d_current, dd_current, nSamples * NMODELS * sizeof(scalar));
        currentSz = nSamples;
    }
}

// Compute the current deviation of both tuned and untuned models against each tuned model
// Models are interleaved (even id = tuned, odd id = detuned) in SamplingProfiler
__global__ void compute_err_and_gradient(unsigned int nSamples, scalar *targetParam, scalar *current, scalar *gradient, scalar *err)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; // probe
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; // reference
    scalar tuned_err = 0, detuned_err = 0;
    for ( unsigned int i = 0; i < nSamples; i++ ) {
        tuned_err += std::fabs(current[2*x + NMODELS*i] - current[2*y + NMODELS*i]);
        detuned_err += std::fabs(current[2*x+1 + NMODELS*i] - current[2*y + NMODELS*i]);
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

extern "C" void doProfile(ProfilerLibrary::Pointers &pointers, size_t target, unsigned int nSamples,double &accuracy, double &median_norm_gradient)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_stim, pointers.stim, sizeof(Stimulation), cudaMemcpyHostToDevice));
    stimPROF = pointers.d_stim;

    resize_current(nSamples);
    currentPROF = d_current;

    stepTimeGPU();

    dim3 block(32, 16);
    dim3 grid(NPAIRS/32, NPAIRS/16);
    compute_err_and_gradient<<<grid, block>>>(nSamples, pointers.d_param[target], d_current, d_gradient, d_err);

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
