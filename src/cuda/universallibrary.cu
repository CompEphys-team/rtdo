#ifndef UNIVERSAL_CU
#define UNIVERSAL_CU

#include "lib_definitions.h"

#include "bubble.cu"
#include "cluster.cu"
#include "deltabar.cu"
#include "deviations.cu"
#include "profile.cu"
#include "util.cu"

static std::vector<cudaStream_t> lib_streams(1, 0);
inline cudaStream_t getLibStream(unsigned int streamId)
{
    unsigned int oldSz = lib_streams.size();
    if ( streamId >= oldSz ) {
        lib_streams.resize(streamId+1);
        for ( unsigned int i = oldSz; i < streamId+1; i++ )
            cudaStreamCreate(&lib_streams[i]);
    }
    return lib_streams[streamId];
}

static std::vector<cudaEvent_t> lib_events;
static unsigned int nextEvent = 0;
inline cudaEvent_t getLibEvent(unsigned int eventHandle)
{
    unsigned int oldSz = lib_events.size();
    if ( eventHandle >= oldSz ) {
        lib_events.resize(eventHandle+1);
        for ( unsigned int i = oldSz; i < eventHandle+1; i++ )
            cudaEventCreate(&lib_events[i]);
    }
    return lib_events[eventHandle];
}

void libInit(UniversalLibrary &lib, UniversalLibrary::Pointers &pointers)
{
    pointers.pushV = [](void *hostptr, void *devptr, size_t size, int streamId){
        if ( streamId < 0 )
            CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostptr, size, cudaMemcpyHostToDevice))
        else
            CHECK_CUDA_ERRORS(cudaMemcpyAsync(devptr, hostptr, size, cudaMemcpyHostToDevice, getLibStream(streamId)))
    };
    pointers.pullV = [](void *hostptr, void *devptr, size_t size, int streamId){
        if ( streamId < 0 )
            CHECK_CUDA_ERRORS(cudaMemcpy(hostptr, devptr, size, cudaMemcpyDeviceToHost))
        else
            CHECK_CUDA_ERRORS(cudaMemcpyAsync(hostptr, devptr, size, cudaMemcpyDeviceToHost, getLibStream(streamId)))
    };

    pointers.target =& target;
    pointers.output =& timeseries;
    pointers.summary =& summary;

    pointers.clusters =& clusters;
    pointers.clusterCurrent =& clusterCurrent;
    pointers.clusterPrimitives =& sections;
    pointers.clusterObs =& clusterObs;

    pointers.bubbles =& bubbles;

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

    CHECK_CUDA_ERRORS(cudaMalloc(&d_prof_error, profSz * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prof_dist_uw, profSz * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prof_dist_to, profSz * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_prof_dist_w, profSz * sizeof(scalar)));

    // Philox is fastest for normal dist, if developer.nvidia.com/curand is to be believed
    CURAND_CALL(curandCreateGenerator(&cuRNG, CURAND_RNG_PSEUDO_PHILOX4_32_10));
}

extern "C" void libExit(UniversalLibrary::Pointers &pointers)
{
    for ( size_t i = 1; i < lib_streams.size(); i++ )
        CHECK_CUDA_ERRORS(cudaStreamDestroy(lib_streams[i]));

    freeMem();
    pointers.pushV = pointers.pullV = nullptr;
    CURAND_CALL(curandDestroyGenerator(cuRNG));

    CHECK_CUDA_ERRORS(cudaFree(d_target));
    CHECK_CUDA_ERRORS(cudaFree(d_timeseries));
    CHECK_CUDA_ERRORS(cudaFree(d_summary));
    CHECK_CUDA_ERRORS(cudaFree(d_prof_error));
    CHECK_CUDA_ERRORS(cudaFree(d_prof_dist_uw));
    CHECK_CUDA_ERRORS(cudaFree(d_prof_dist_to));
    CHECK_CUDA_ERRORS(cudaFree(d_prof_dist_w));
    CHECK_CUDA_ERRORS(cudaFree(d_random));
    CHECK_CUDA_ERRORS(cudaFree(d_clusters));
    CHECK_CUDA_ERRORS(cudaFree(d_clusterLen));
    CHECK_CUDA_ERRORS(cudaFree(d_clusterMasks));
    CHECK_CUDA_ERRORS(cudaFree(d_clusterCurrent));
    CHECK_CUDA_ERRORS(cudaFree(d_sections));
    CHECK_CUDA_ERRORS(cudaFree(d_currents));
    CHECK_CUDA_ERRORS(cudaFree(d_clusterObs));
    CHECK_CUDA_ERRORS(cudaFree(d_bubbles));

    CHECK_CUDA_ERRORS(cudaFreeHost(target));
    CHECK_CUDA_ERRORS(cudaFreeHost(timeseries));
    CHECK_CUDA_ERRORS(cudaFreeHost(summary));
    CHECK_CUDA_ERRORS(cudaFreeHost(clusters));
    CHECK_CUDA_ERRORS(cudaFreeHost(clusterLen));
    CHECK_CUDA_ERRORS(cudaFreeHost(clusterCurrent));
    CHECK_CUDA_ERRORS(cudaFreeHost(sections));
    CHECK_CUDA_ERRORS(cudaFreeHost(clusterObs));
    CHECK_CUDA_ERRORS(cudaFreeHost(bubbles));
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

extern "C" void pushTarget(int streamId, size_t nSamples, size_t offset)
{
    if ( nSamples == 0 )
        nSamples = latest_target_size;
    if ( streamId < 0 )
        CHECK_CUDA_ERRORS(cudaMemcpy(d_target+offset, target+offset, nSamples * sizeof(scalar), cudaMemcpyHostToDevice))
    else
        CHECK_CUDA_ERRORS(cudaMemcpyAsync(d_target+offset, target+offset, nSamples * sizeof(scalar), cudaMemcpyHostToDevice, getLibStream(streamId)))
}

extern "C" void resizeOutput(size_t newSize)
{
    resizeArrayPair(timeseries, d_timeseries, timeseries_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_timeseries, &d_timeseries, sizeof(scalar*)));
    latest_timeseries_size = newSize;
}

extern "C" void pullOutput(int streamId)
{
    if ( streamId < 0 )
        CHECK_CUDA_ERRORS(cudaMemcpy(timeseries, d_timeseries, latest_timeseries_size * sizeof(scalar), cudaMemcpyDeviceToHost))
    else
        CHECK_CUDA_ERRORS(cudaMemcpyAsync(timeseries, d_timeseries, latest_timeseries_size * sizeof(scalar), cudaMemcpyDeviceToHost, getLibStream(streamId)))
}

extern "C" void resizeSummary(size_t newSize)
{
    resizeArrayPair(summary, d_summary, summary_size, newSize);
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_summary, &d_summary, sizeof(scalar*)));
    latest_summary_size = newSize;
}

extern "C" void pullSummary(int streamId, size_t nSamples, size_t offset)
{
    if ( nSamples == 0 )
        nSamples = latest_summary_size;
    if ( streamId < 0 )
        CHECK_CUDA_ERRORS(cudaMemcpy(summary+offset, d_summary+offset, nSamples * sizeof(scalar), cudaMemcpyHostToDevice))
    else
        CHECK_CUDA_ERRORS(cudaMemcpyAsync(summary+offset, d_summary+offset, nSamples * sizeof(scalar), cudaMemcpyHostToDevice, getLibStream(streamId)))
}

extern "C" void libSync(unsigned int streamId)
{
    if ( streamId )
        CHECK_CUDA_ERRORS(cudaStreamSynchronize(getLibStream(streamId)))
    else
        CHECK_CUDA_ERRORS(cudaDeviceSynchronize())
}

extern "C" void libResetEvents(unsigned int nExpected)
{
    nextEvent = 0;
    getLibEvent(nExpected);
}

extern "C" unsigned int libRecordEvent(unsigned int streamId)
{
    cudaEventRecord(getLibEvent(nextEvent), getLibStream(streamId));
    return nextEvent++;
}

extern "C" void libWaitEvent(unsigned int eventHandle, unsigned int streamId)
{
    cudaStreamWaitEvent(getLibStream(streamId), getLibEvent(eventHandle), 0);
}

#endif
