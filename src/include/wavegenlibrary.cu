#ifndef WAVEGEN_CU
#define WAVEGEN_CU

#include "wavegenlibrary.h"
#include "cuda_helper.h" // For syntax highlighting only
#include <thrust/device_vector.h>
#include <thrust/extrema.h>

static __device__ WaveStats *dd_wavestats;
static __device__ Stimulation *dd_waveforms;

static __device__ scalar dd_err[MM_NumModels];
static scalar err[MM_NumModels];

static unsigned int obsValuesSz = 1024 * MM_NumGroups;

static unsigned int obsFitnessSz = 0;
static scalar *obsFitness;

void allocateGroupMem(WavegenLibrary::Pointers &pointers)
{
    cudaHostAlloc(&pointers.wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
        deviceMemAllocate(&pointers.d_wavestats, dd_wavestats, MM_NumGroups * sizeof(WaveStats));
    cudaHostAlloc(&pointers.waveforms, MM_NumGroups * sizeof(Stimulation), cudaHostAllocPortable);
        deviceMemAllocate(&pointers.d_waveforms, dd_waveforms, MM_NumGroups * sizeof(Stimulation));

    cudaHostAlloc(&pointers.clear_wavestats, MM_NumGroups * sizeof(WaveStats), cudaHostAllocPortable);
    for ( unsigned i = 0; i < MM_NumGroups; i++ )
        pointers.clear_wavestats[i] = {};
}

void freeGroupMem(WavegenLibrary::Pointers &pointers)
{
    cudaFreeHost(pointers.wavestats);
    cudaFreeHost(pointers.clear_wavestats);
    CHECK_CUDA_ERRORS(cudaFree(pointers.d_wavestats));
    cudaFreeHost(pointers.waveforms);
    CHECK_CUDA_ERRORS(cudaFree(pointers.d_waveforms));
}

void libInit(WavegenLibrary::Pointers &pointers, size_t numGroups, size_t numModels)
{
    pointers.clearStats = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_wavestats, pointers.clear_wavestats, numGroups * sizeof(WaveStats), cudaMemcpyHostToDevice))
    };
    pointers.pullStats = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.wavestats, pointers.d_wavestats, numGroups * sizeof(WaveStats), cudaMemcpyDeviceToHost))
    };
    pointers.pushWaveforms = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_waveforms, pointers.waveforms, numGroups * sizeof(Stimulation), cudaMemcpyHostToDevice))
    };
    pointers.pullWaveforms = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.waveforms, pointers.d_waveforms, numGroups * sizeof(Stimulation), cudaMemcpyDeviceToHost))
    };
    pointers.pushErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_err, pointers.err, numModels * sizeof(scalar)))
    };
    pointers.pullErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(pointers.err, dd_err, numModels * sizeof(scalar)))
    };
    pointers.err = err;

    allocateMem();
    allocateGroupMem(pointers);
    initialize();
    pointers.clearStats();
}
void libInitPost(WavegenLibrary::Pointers &pointers)
{
    CHECK_CUDA_ERRORS(cudaMalloc(pointers.findObsValues, obsValuesSz * MM_NumGroups * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMalloc(&obsFitness, obsFitnessSz * MM_NumGroups * sizeof(scalar)));
}

extern "C" void libExit(WavegenLibrary::Pointers &pointers)
{
    freeMem();
    freeGroupMem(pointers);
    CHECK_CUDA_ERRORS(cudaFree(*pointers.findObsValues));
    CHECK_CUDA_ERRORS(cudaFree(obsFitness));
    pointers.clearStats = pointers.pullStats = pointers.pushWaveforms = pointers.pullWaveforms = pointers.pushErr = pointers.pullErr = nullptr;
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

template <typename T>
void resizeArray(T *&arr, unsigned int &actualSize, unsigned int requestedSize)
{
    if ( actualSize < requestedSize ) {
        CHECK_CUDA_ERRORS(cudaFree(arr));
        CHECK_CUDA_ERRORS(cudaMalloc(&arr, requestedSize * sizeof(T)));
        actualSize = requestedSize;
    }
}

// Code from Justin Luitjens, <https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/>
// Note, shuffle is only supported on compute capability 3.x and higher
// Code from Justin Luitjens, see above
__device__ inline scalar warpReduceSum(scalar val)
{
    for (int offset = warpSize/2; offset > 0; offset /= 2)
        val += __shfl_down(val, offset);
    return val;
}

// Code from Justin Luitjens, see above
__device__ inline scalar blockReduceSum(scalar val)
{
    static __shared__ int shared[32]; // Shared mem for 32 partial sums
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = warpReduceSum(val);     // Each warp performs partial reduction

    if (lane==0) shared[wid]=val; // Write reduced value to shared memory

    __syncthreads();              // Wait for all partial reductions

    //read from shared memory only if that warp existed
    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid==0) val = warpReduceSum(val); //Final reduce within first warp

    return val;
}

__global__ void reduceKernel(scalar *in, scalar *out, unsigned int nGroupSums)
{
    // global x matches pre-reduced group sum
    // global y/z matches values y/z, i.e. starts/ends
    scalar sum = 0;

    //reduce multiple elements per thread
    for ( int x = blockIdx.x * blockDim.x + threadIdx.x;
         x < nGroupSums;
         x += blockDim.x * gridDim.x) {
        sum += in[x + nGroupSums*blockIdx.y + nGroupSums*gridDim.y*blockIdx.z];
    }

    sum = blockReduceSum(sum);
    if ( threadIdx.x == 0 )
        out[blockIdx.x + nGroupSums*blockIdx.y + nGroupSums*gridDim.y*blockIdx.z] = sum;
}

__global__ void getObsFitness(scalar *values, scalar *out, unsigned int nSamples, unsigned int nEnds)
{
    const unsigned int group = threadIdx.x + blockDim.x*blockIdx.x;
    const unsigned int startOffset = blockIdx.y;
    scalar sum = 0;
    scalar fitness;
    unsigned int i = startOffset;

    // Accumulate samples from the assigned start to just before the first end
    if ( group < MM_NumGroups )
        for ( unsigned int beforeFirstEnd = nSamples - nEnds; i < beforeFirstEnd; i++ )
            sum += values[i*MM_NumGroups + group];

    // Accumulate remaining samples
    for ( unsigned int endIdx = 0; i < nSamples; i++, endIdx++ ) {
        if ( group < MM_NumGroups ) // Check in loop, as  all threads must participate in blockReduceSum
            sum += values[i*MM_NumGroups + group];

        // Compute fitness of the obs from the assigned start to the present sample, which is an end
        fitness = sum / (i-startOffset);

        // Sum across all groups within the block, write intermediate output
        // Note, this acts as a first pass of reduceKernel invocation.
        fitness = blockReduceSum(fitness);
        if ( threadIdx.x == 0 ) {
            // x: blocks     y: starts               z: ends
            out[blockIdx.x + gridDim.x*startOffset + gridDim.x*gridDim.y*endIdx] = fitness;
        }
    }
}

extern "C" void findObservationWindow(WavegenLibrary::Pointers &pointers,
                                      Stimulation &stim,
                                      unsigned int nStart,
                                      unsigned int nEnd,
                                      unsigned int nSamples,
                                      scalar cycleDuration)
{
    resizeArray(*pointers.findObsValues, obsValuesSz, nSamples * MM_NumGroups);
    CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_waveforms, &stim, sizeof(Stimulation), cudaMemcpyHostToDevice));

    // Simulate, recording fitnessPartial at each cycle within stim.tObs
    stepTimeGPU();

    // Calculate fitness (mean fitnessPartial) for all nStart x nEnd possible obs
    // Also does a first reduce pass to keep obsFitness reasonably sized at gridSz.x * starts * ends
    int blockSz, minGridSz;
    cudaOccupancyMaxPotentialBlockSize(&minGridSz, &blockSz, getObsFitness, 0, MM_NumGroups*nStart);
    dim3 calcGridSz = dim3((MM_NumGroups + blockSz - 1)/blockSz, nStart);
    resizeArray(obsFitness, obsFitnessSz, calcGridSz.x * nStart * nEnd);
    getObsFitness<<<calcGridSz, blockSz>>>(*pointers.findObsValues, obsFitness, nSamples, nEnd);

    // Reduce to single starts*ends
    thrust::device_vector<scalar> sums(nStart * nEnd);
    blockSz = 512;
    dim3 reduceGridSz = dim3(std::min((calcGridSz.x + blockSz - 1) / blockSz, 1024u), nStart, nEnd);
    if ( reduceGridSz.x > 1 ) {
        thrust::device_vector<scalar> tmp(reduceGridSz.x * nStart * nEnd);
        reduceKernel<<<reduceGridSz, blockSz>>>(*pointers.findObsValues, tmp.data().get(), calcGridSz.x);
        reduceKernel<<<dim3(1, reduceGridSz.y, reduceGridSz.z), blockSz>>>(tmp.data().get(), sums.data().get(), reduceGridSz.x);
    } else {
        reduceKernel<<<reduceGridSz, blockSz>>>(*pointers.findObsValues, sums.data().get(), calcGridSz.x);
    }

    // Find greatest sum fitness
    unsigned int maxIdx = thrust::max_element(sums.begin(), sums.end()) - sums.begin();
    // since maxIdx == bestStart + nStart * bestEnd:
    unsigned int bestStart = maxIdx % nStart;
    unsigned int bestEnd = maxIdx / nStart;
    stim.tObsBegin += bestStart * cycleDuration;
    stim.tObsEnd -= (nEnd - bestEnd - 1) * cycleDuration;
}

#endif
