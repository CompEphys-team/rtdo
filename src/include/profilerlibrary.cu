#ifndef PROFILER_CU
#define PROFILER_CU

#include "profilerlibrary.h"
#include "cuda_helper.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/count.h>

static scalar *d_gradient, *d_err;
static unsigned int currentSz = 1024;
static constexpr unsigned int gradientSz = NCOMP - NPAIRS; // No diagonal
static constexpr unsigned int errSz = (NCOMP - NPAIRS)/2; // Above diagonal only

__constant__ iStimulation stim;

extern "C" void pushStim(const iStimulation &h_stim)
{
    cudaMemcpyToSymbol(stim, &h_stim, sizeof(iStimulation));
}

void libInit(ProfilerLibrary::Pointers &pointers)
{
    CHECK_CUDA_ERRORS(cudaMalloc(&d_gradient, gradientSz * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_err, errSz * sizeof(scalar)));

    allocateMem();
    initialize();
}
void libInitPost(ProfilerLibrary::Pointers &pointers)
{
    CHECK_CUDA_ERRORS(cudaMalloc(pointers.current, currentSz * NMODELS * sizeof(scalar)));
}

extern "C" void libExit(ProfilerLibrary::Pointers &pointers)
{
    freeMem();
    CHECK_CUDA_ERRORS(cudaFree(*pointers.current));
    CHECK_CUDA_ERRORS(cudaFree(d_gradient));
    CHECK_CUDA_ERRORS(cudaFree(d_err));
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

void resize_current(unsigned int nSamples, ProfilerLibrary::Pointers &pointers)
{
    if ( currentSz < nSamples ) {
        CHECK_CUDA_ERRORS(cudaFree(*pointers.current));
        CHECK_CUDA_ERRORS(cudaMalloc(pointers.current, nSamples * NMODELS * sizeof(scalar)));
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
    resize_current(nSamples, pointers);

    stepTimeGPU();

    dim3 block(32, 16);
    dim3 grid(NPAIRS/32, NPAIRS/16);
    compute_err_and_gradient<<<grid, block>>>(nSamples, pointers.d_param[target], *pointers.current, d_gradient, d_err);

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
