#include "lib_definitions.h"

void pushDeltabar(std::vector<double> dbar)
{
    scalar h_deltabar[NPARAMS];
    for ( int i = 0; i < NPARAMS; i++ )
        h_deltabar[i] = dbar[i];
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(deltabar, h_deltabar, NPARAMS*sizeof(scalar)));
}

std::vector<unsigned short> pushDetuneIndices(int trajLen, int nTraj, const MetaModel &model)
{
    std::vector<int> detuneIndices = model.get_detune_indices(trajLen, nTraj);
    std::vector<unsigned char> h_detuneParamIndices(detuneIndices.size());
    std::vector<unsigned short> nDetunes(NPARAMS);
    for ( int i = 0; i < NPARAMS; i++ )
        nDetunes[i] = 0;
    for ( size_t i = 0; i < detuneIndices.size(); i++ ) {
        if ( detuneIndices[i] >= 0 )
            ++nDetunes[detuneIndices[i]];
        h_detuneParamIndices[i] = detuneIndices[i]; // Note, the negative indices are never consumed, so unsigned is not an error.
    }
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(detuneParamIndices, h_detuneParamIndices.data(), detuneIndices.size() * sizeof(unsigned char)));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(numDetunesByParam, nDetunes.data(), NPARAMS * sizeof(unsigned short)));
    return nDetunes;
}



__global__ void observe_no_steps_kernel(int blankCycles)
{
    unsigned int id = blockDim.x * blockIdx.x + threadIdx.x;
    if ( id >= NMODELS )
        return;
    iStimulation stim = dd_stimUNI[id];
    iObservations obs = {};
    int tStart = 0;
    int nextObs = 0;
    if ( blankCycles > 0 ) {
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



extern "C" void genRandom(unsigned int n, scalar mean, scalar sd, unsigned long long seed)
{
    resizeArray(d_random, random_size, n * sizeof(scalar));
    CHECK_CUDA_ERRORS(cudaMemcpyToSymbol(dd_random, &d_random, sizeof(scalar*)));

    if ( seed != 0 )
        CURAND_CALL(curandSetPseudoRandomGeneratorSeed(cuRNG, seed));

#ifdef USEDOUBLE
    CURAND_CALL(curandGenerateNormalDouble(cuRNG, d_random, n, mean, sd));
#else
    CURAND_CALL(curandGenerateNormal(cuRNG, d_random, n, mean, sd));
#endif

}
