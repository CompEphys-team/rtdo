#ifndef WAVEGEN_CU
#define WAVEGEN_CU

#include "wavegenlibrary.h"
#include "cuda_helper.h" // For syntax highlighting only

#define TARGET_DIAG -2

static __device__ iStimulation *dd_waveforms;

static __device__ scalar dd_err[MM_NumModels];
static scalar err[MM_NumModels];

static scalar *parfitBlock, *parfitStim;
static unsigned int parfitBlockSz, parfitStimSz;
static __constant__ scalar *dd_parfitBlock;

static Bubble *bubblesTemp = nullptr, *bubblesFinal = nullptr;
static unsigned int bubblesTempSz = 0, bubblesFinalSz = 0;
static __constant__ Bubble *dd_bubbles;

static __device__ scalar *dd_diagDelta;
static unsigned int diagDeltaSz = 0;

void allocateGroupMem(WavegenLibrary::Pointers &pointers)
{
    cudaHostAlloc(&pointers.waveforms, MM_NumGroups * sizeof(iStimulation), cudaHostAllocPortable);
        deviceMemAllocate(&pointers.d_waveforms, dd_waveforms, MM_NumGroups * sizeof(iStimulation));

    CHECK_CUDA_ERRORS(cudaHostAlloc(&pointers.bubbles, MM_NumGroups * sizeof(Bubble), cudaHostAllocPortable));
    pointers.d_bubbles = nullptr;
}

void freeGroupMem(WavegenLibrary::Pointers &pointers)
{
    cudaFreeHost(pointers.waveforms);
    CHECK_CUDA_ERRORS(cudaFree(pointers.d_waveforms));
}

void libInit(WavegenLibrary::Pointers &pointers, size_t numGroups, size_t numModels)
{
    pointers.pullBubbles = [&pointers](unsigned int n){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.bubbles, pointers.d_bubbles, n * sizeof(Bubble), cudaMemcpyDeviceToHost))
    };
    pointers.pushWaveforms = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.d_waveforms, pointers.waveforms, numGroups * sizeof(iStimulation), cudaMemcpyHostToDevice))
    };
    pointers.pullWaveforms = [&pointers, numGroups](){
        CHECK_CUDA_ERRORS(cudaMemcpy(pointers.waveforms, pointers.d_waveforms, numGroups * sizeof(iStimulation), cudaMemcpyDeviceToHost))
    };
    pointers.pushErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_err, pointers.err, numModels * sizeof(scalar)))
    };
    pointers.pullErr = [&pointers, numModels](){
        CHECK_CUDA_ERRORS(cudaMemcpyFromSymbol(pointers.err, dd_err, numModels * sizeof(scalar)))
    };
    pointers.err = err;

    pointers.diagDelta = pointers.d_diagDelta = nullptr;

    allocateMem();
    allocateGroupMem(pointers);
    initialize();
}

extern "C" void libExit(WavegenLibrary::Pointers &pointers)
{
    freeMem();
    freeGroupMem(pointers);
    cudaFreeHost(pointers.bubbles);
    cudaFree(parfitBlock);
    cudaFree(parfitStim);
    cudaFree(bubblesTemp);
    cudaFree(bubblesFinal);
    // pointers.d_bubbles points at either of the static bubbles*
    pointers.pullErr = pointers.pushErr = pointers.pullWaveforms = pointers.pushWaveforms = nullptr;
    pointers.pullBubbles = nullptr;

    cudaFree(pointers.d_diagDelta);
    cudaFreeHost(pointers.diagDelta);
    pointers.diagnose = nullptr;
}

extern "C" void resetDevice()
{
    cudaDeviceReset();
}

__device__ inline Bubble shuffle_down(Bubble b, int offset)
{
    return Bubble {
        __shfl_down(b.startCycle, offset),
        __shfl_down(b.cycles, offset),
        __shfl_down(b.value, offset)
    };
}

__device__ void warpMergeBubbles(Bubble start, Bubble mid, Bubble end, int cyclesPerTriplet, int cutoff, Bubble *out, bool final)
{
    Bubble nextStart, nextMid, nextEnd;
    for ( int offset = 1; offset < cutoff; offset *= 2 ) {
        // Shuffle down
        nextStart = shuffle_down(start, offset);
        nextMid = shuffle_down(mid, offset);
        nextEnd = shuffle_down(end, offset);

        // Merge across the seam
        if ( nextStart.cycles ) {
            if ( start.cycles == cyclesPerTriplet*offset ) { // full-length merge
                // Merge nextStart into start; invalidate nextStart
                start.cycles += nextStart.cycles;
                start.value += nextStart.value;
                nextStart.cycles = 0;
            } else if ( end.cycles ) { // end merge
                if ( nextStart.cycles == cyclesPerTriplet*offset ) { // nextStart is full-length
                    // Merge end and nextStart, save to new end, invalidate end and nextStart
                    nextEnd = {end.startCycle,
                               end.cycles+nextStart.cycles,
                               end.value+nextStart.value};
                    end.cycles = 0;
                } else {
                    // Merge nextStart into end; invalidate nextStart
                    end.cycles += nextStart.cycles;
                    end.value += nextStart.value;
                }
                nextStart.cycles = 0;
            } else if ( nextStart.cycles == cyclesPerTriplet*offset ) { // No merge, nextStart is new end
                nextEnd = nextStart;
                nextStart.cycles = 0;
            }
        }

        // Replace mid with the largest of (mid, end, nextStart, nextMid)
        scalar midFitness = mid.cycles ? mid.value/mid.cycles : 0;
        if ( end.cycles && end.value/end.cycles > midFitness )
            mid = end;
        if ( nextStart.cycles && nextStart.value/nextStart.cycles > midFitness )
            mid = nextStart;
        if ( nextMid.cycles && nextMid.value/nextMid.cycles > midFitness )
            mid = nextMid;

        // Replace end with nextEnd
        end = nextEnd;
    }

    // Save from lane 0
    if ( threadIdx.x % warpSize == 0 ) {
        if ( final ) {
            // Finalise value to fitness, save only the best
            start.value = start.cycles ? start.value/start.cycles : 0;
            mid.value = mid.cycles ? mid.value/mid.cycles : 0;
            end.value = end.cycles ? end.value/end.cycles : 0;
            if ( start.value > mid.value && start.value > end.value )
                out[0] = start;
            else if ( mid.value > end.value )
                out[0] = mid;
            else
                out[0] = end;
        } else {
            out[0] = start;
            out[1] = mid;
            out[2] = end;
        }
    }
}

__global__ void deviceProcessBubbles(void *in, Bubble *out, int N, int cyclesPerInputTriplet, scalar norm, bool final)
{
    __shared__ Bubble bubbles[3*32];
    int warpCutoff = ((N + warpSize - 1)/warpSize)*warpSize;
    int x  = blockIdx.x * blockDim.x + threadIdx.x;
    Bubble start, mid, end;

    // Fully engage all warps that have any work to do
    if ( x < warpCutoff ) {
        if ( x < N ) {
            // Create bubbles in the first place...
            if ( cyclesPerInputTriplet == 1 ) {
                // Existing samples open a bubble at cycle x, lasting 0 or 1 cycle, depending on whether it meets the threshold
                // mid/end stay empty, as does start for non-existing samples
                scalar val = ((scalar*)in)[x + blockIdx.y*N] / norm;
                start = {x, val>1, val};
                mid = end = {0,0,0};
            }
            // ... or merge existing ones
            else {
                start = ((Bubble*)in)[3*(x + N*blockIdx.y)];
                mid = ((Bubble*)in)[3*(x + N*blockIdx.y) + 1];
                end = ((Bubble*)in)[3*(x + N*blockIdx.y) + 2];
            }
        } else {
            start = mid = end = {0,0,0};
        }
        warpMergeBubbles(start, mid, end, cyclesPerInputTriplet, warpSize, bubbles + 3*(threadIdx.x/warpSize), false);
    }

    // Sync within block to ensure __shared__ bubbles is fully populated
    __syncthreads();

    // Fully engage first warp to reduce to one block-level bubble triplet
    if ( threadIdx.x < warpSize ) {
        // Let the i:th thread in warp 0 target the first thread of the i:th warp;
        // threads whose targets did not create bubbles must participate with empty bubbles
        if ( threadIdx.x < blockDim.x/warpSize && x-threadIdx.x + threadIdx.x*warpSize < warpCutoff ) {
            start = bubbles[threadIdx.x*3];
            mid = bubbles[threadIdx.x*3 + 1];
            end = bubbles[threadIdx.x*3 + 2];
        } else {
            start = mid = end = {0,0,0};
        }
        warpMergeBubbles(start, mid, end,
                         warpSize*cyclesPerInputTriplet, blockDim.x/warpSize,
                         out + (final ? blockIdx.y : (3*(blockIdx.x + gridDim.x*blockIdx.y))),
                         final);
    }
}

// Code adapated from Justin Luitjens, <https://devblogs.nvidia.com/parallelforall/faster-parallel-reductions-kepler/>
// Note, shuffle is only supported on compute capability 3.x and higher
__device__ inline scalar warpReduceSum(scalar val, int cutoff = warpSize)
{
    for ( int offset = 1; offset < cutoff; offset *= 2 )
        val += __shfl_down(val, offset);
    return val;
}

// Code adapted from Justin Luitjens, see above
__device__ inline scalar blockReduceSum(scalar val)
{
    static __shared__ scalar shared[32]; // Shared mem for 32 partial sums
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

__global__ void deviceReduceSum_block2stim(scalar *in, scalar *out, unsigned int nBlocksPerStim, unsigned int nSamples)
{
    /** Input structure:                Output structure:
     *    W1        W2        ...          t1 t2 ...
     *    B1 ... Bn B1 ... Bn ...       W1
     * t1                               W2
     * t2                               ..
     * ..
     *
     * Launch structure:
     * x (threads, 1 block)        (B): GeNN block idx relative to stim
     * y (blocks, full-width grid) (W): stim idx
     * z (blocks, stride grid)     (t): time
     * */

    for ( unsigned int t = blockIdx.z; t < nSamples; t += gridDim.z ) {
        scalar sum = 0;

        //reduce multiple elements per thread
        for ( int x = threadIdx.x; x < nBlocksPerStim; x += blockDim.x ) {
            sum += in[x + nBlocksPerStim*blockIdx.y + nBlocksPerStim*gridDim.y*t];
        }

        sum = blockReduceSum(sum);
        if ( threadIdx.x == 0 )
            out[t + nSamples*blockIdx.y] = sum;
    }
}

extern "C" void generateBubbles(unsigned int nSamples, unsigned int nStim, WavegenLibrary::Pointers &pointers)
{
    const unsigned int nGroupsPerStim = MM_NumGroups/nStim;
    const unsigned int nBlocksPerStim = nGroupsPerStim/MM_NumGroupsPerBlock;
    unsigned int blockSz = 512;
    void *devSymPtr;

    if ( nGroupsPerStim <= MM_NumGroupsPerBlock ) {
        // Prepare bubbles array
        resizeArray(bubblesFinal, bubblesFinalSz, nStim);
        CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymPtr, dd_bubbles));
        CHECK_CUDA_ERRORS(cudaMemcpy(devSymPtr, &bubblesFinal, sizeof(void*), cudaMemcpyHostToDevice));
    } else {
        // Prepare per-block sample array
        resizeArray(parfitBlock, parfitBlockSz, MM_NumBlocks * nSamples);
        CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymPtr, dd_parfitBlock));
        CHECK_CUDA_ERRORS(cudaMemcpy(devSymPtr, &parfitBlock, sizeof(void*), cudaMemcpyHostToDevice));
    }

    *pointers.nGroupsPerStim = nGroupsPerStim;

    // Stimulate
    stepTimeGPU();

    if ( nGroupsPerStim <= MM_NumGroupsPerBlock ) {
        // All reduction and bubble creation happens in the GeNN kernel; nothing else to do.
        pointers.d_bubbles = bubblesFinal;
        return;
    }

    // Reduce/transpose samples from one column per GeNN block to one row per stim
    // Prepare per-stim sample array
    resizeArray(parfitStim, parfitStimSz, nStim * nSamples);

    // Reduce per-group to per-stim samples -- see notes in deviceReduceSum_group2stim
    blockSz = std::min(512, 32*int((nBlocksPerStim+31)/32));
    deviceReduceSum_block2stim<<< dim3(1, nStim, 1024), blockSz >>>(parfitBlock, parfitStim, nBlocksPerStim, nSamples);

    // Prepare bubble arrays
    blockSz = 512;
    unsigned int nTriplets = ((nSamples + blockSz - 1)/blockSz);
    unsigned int nTripletsNext = ((nTriplets + blockSz - 1)/blockSz);
    resizeArray(bubblesTemp, bubblesTempSz, nStim * nTriplets * 3);
    resizeArray(bubblesFinal, bubblesFinalSz, nStim * nTripletsNext * 3);

    // Create bubbles (immediately reducing to one triplet per block)
    deviceProcessBubbles<<< dim3(nTriplets, nStim), blockSz >>>(parfitStim, bubblesTemp, nSamples, 1, nGroupsPerStim, false);

    // Reduce bubbles to a single, final bubble per stim
    // Multiple reductions may be required to get to one triplet per stim
    // Exponentially ratchet down to single triplets
    unsigned int cycles = blockSz;
    bool reverseSwap = false;
    while ( nTriplets > 1 /* entry condition only */ ) {
        deviceProcessBubbles<<< dim3(nTripletsNext, nStim), blockSz >>>(bubblesTemp, bubblesFinal, nTriplets, cycles, 0, nTripletsNext==1);
        if ( nTripletsNext == 1 )
            break;
        cycles *= blockSz;
        nTriplets = nTripletsNext;
        nTripletsNext = ((nTriplets + blockSz - 1)/blockSz);
        std::swap(bubblesTemp, bubblesFinal);
        reverseSwap = !reverseSwap;
    }

    // Point to final bubble array
    pointers.d_bubbles = bubblesFinal;

    // Clean up
    if ( reverseSwap )
        std::swap(bubblesTemp, bubblesFinal);
}

extern "C" void diagnose(WavegenLibrary::Pointers &pointers, iStimulation I)
{
    unsigned int tmp = diagDeltaSz, reqSz = I.duration * (NPARAM+1);
    resizeArray(pointers.d_diagDelta, diagDeltaSz, reqSz);
    resizeHostArray(pointers.diagDelta, tmp, reqSz);

    void *devSymPtr;
    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devSymPtr, dd_diagDelta));
    CHECK_CUDA_ERRORS(cudaMemcpy(devSymPtr, &pointers.d_diagDelta, sizeof(void*), cudaMemcpyHostToDevice));

    pointers.waveforms[0] = I;
    pointers.pushWaveforms();
    *pointers.targetParam = TARGET_DIAG;
    *pointers.getErr = true;

    stepTimeGPU();

    CHECK_CUDA_ERRORS(cudaMemcpy(pointers.diagDelta, pointers.d_diagDelta, reqSz * sizeof(scalar), cudaMemcpyDeviceToHost))
}

#endif
