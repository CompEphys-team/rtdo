/*--------------------------------------------------------------------------
Author: Daniel Saska

Institute: Informatics
University of Sussex
Brighton BN1 9QJ, UK

email to:  ds376@sussex.ac.uk

initial version: 2014-09-09

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file VClampGA.cu

\brief Main entry point for the GeNN project demonstrating realtime fitting of a neuron with a GA running mostly on the GPU.
*/
//--------------------------------------------------------------------------


// minimum duration of voltage step
#define MINSTEP 2.0  
// minimal voltage
#define MINV -100.0
// maximal voltage
#define MAXV 50.0
// minimal observation time window
#define MINT 5.0
// maximal observation time window
#define MAXT 100.0
// observation time (fixed)
#define OT 100.0
// total simulation time
#define SIM_TIME 200.0
// initial dispersion of V steps
#define VSTEPINI 40.0
// initial "baseline" for V steps
#define VSTEP0 -60.0
// initial maximum width of steps (ms)
#define STEPWDINI 100.0
// minimal position of first step
#define MINSTEPT 10.0
// factor of reducing mutateA
#define MUTATEFAC 0.995
// number of steps in the voltage signals
#define NVSTEPS 3

#define TOTALT 200

// the size of random mutations
double mutateA = 10.0;

// Novelty search param
double maxSigmaToRange = 0.1;
double pertFac = 0.1;
double noveltyThreshold = 10.0;
double optimiseInitProportion = 0.2;


#include "WaveGA.h"
#include "waveHelper.h"
#include <cuda.h>
#include <array>

#include "wavegenNS.h"

class WavegenNS : public WavegenNSVirtual
{
public:
    WavegenNS();
    ~WavegenNS() {}

    void runAll(int nGenerationsNS, int nGenerationsOptimise, std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag);
    void adjustSigmas();
    void noveltySearch(int nGenerations, bool *stopFlag);
    void optimiseAll(int nGenerations, std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag);
    void optimise(int param, int nGenerations, bool *stopFlag);
    void validate(inputSpec &stim, int param, std::ostream &currentfile);

private:
    vector<inputSpec> stims;
    scalar holdingVar[NVAR], singleParamIni[NPARAM];
    array<vector<noveltyBundle>, NPARAM> noveltyDB;

    NNmodel model;
};

extern "C" WavegenNSVirtual *WavegenCreate()
{
    return new WavegenNS();
}

extern "C" void WavegenDestroy(WavegenNSVirtual **_this)
{
    delete *_this;
    *_this = NULL;
}


WavegenNS::WavegenNS() :
    WavegenNSVirtual()
{
    // build the neuronal circuitery
    modelDefinition( model );
    allocateMem();
    initialize();
    rtdo_init_bridge();

    for ( int i = 0; i < NPARAM; i++ ) {
        sigmaAdjust[i] = 1;
    }

    // Get steady-state variable values at holding potential
    for ( int i = 0; i < NVAR; i++ )
        holdingVar[i] = mvar[i][0];
    for ( int i = 0; i < NPARAM; i++ )
        singleParamIni[i] = mparam[i][0];
    for ( double t = 0.0; t < 10000.0; t += DT ) {
        simulateSingleNeuron(holdingVar, singleParamIni, VSTEP0);
    }
}

void WavegenNS::runAll(int nGenerationsNS, int nGenerationsOptimise, ostream &wavefile, ostream &currentfile, bool *stopFlag)
{
    adjustSigmas();
    noveltySearch(nGenerationsNS, stopFlag);
    optimiseAll(nGenerationsOptimise, wavefile, currentfile, stopFlag);
}

void WavegenNS::adjustSigmas()
{
    unsigned int VSize = NPOP*theSize( model.ftype );
    
    // Find maximal sigma values
    double sigmax[NPARAM];
    for ( int j = 0; j < NPARAM; j++ ) {
        if ( aParamPMult[j] )
            sigmax[j] = maxSigmaToRange; // Range-dependent values are problematic because of zero range end points
        else
            sigmax[j] = maxSigmaToRange * (aParamRange[2*j + 1] - aParamRange[2*j]);
    }
    
    stims.clear();
    wave_pop_init( stims, GAPOP );

    // Stage: Adjust sigmas to detune with similar results:
    cout << "Adjusting parameter sigmas..." << endl;
    for ( int r = 0; r < 2; r++ ) {
        otHH = OT;
        oteHH = TOTALT;
        stageHH = stDetuneAdjust;
        reset(holdingVar, pertFac);
        size_t sn[GAPOP] = {};
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU(t);
            for (size_t i = 0; i < GAPOP; ++i) {
                if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                {
                    for (size_t j = 0; j < NPARAM + 1; ++j) {
                        float tmp = stims[i].V[sn[i]];
                        CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                    }
                    ++sn[i];
                }
            }
        }

        CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );

        double pErr[NPARAM] = {}, errTotal = 0.0;
        for ( size_t i = 0; i < GAPOP; ++i ) {
            for ( size_t j = 1; j < NPARAM + 1; ++j ) {
                pErr[j-1] += errHH[i * (NPARAM + 1) + j];
                errTotal += errHH[i * (NPARAM + 1) + j];
            }
        }
        errTotal /= NPARAM;
        double globalAdjust = 1.0;
        for ( size_t j = 0; j < NPARAM; ++j ) {
            double adjustment = errTotal / pErr[j];
            sigmaAdjust[j] *= adjustment;
            double reduce = aParamSigma[j] * sigmaAdjust[j]/sigmax[j];
            if ( reduce > globalAdjust ) {
                globalAdjust = reduce;
            }
        }
        for ( int j = 0; j < NPARAM; j++ ) {
            sigmaAdjust[j] /= globalAdjust;
        }
    }
    for ( int i = 0; i < NPARAM; i++ ) {
        cout << "Parameter " << (i+1) << " sigma adjustment: " << sigmaAdjust[i] << endl;
    }
    cout << endl;
}

void WavegenNS::noveltySearch(int nGenerations, bool *stopFlag)
{
    unsigned int VSize = NPOP*theSize( model.ftype );
    for ( int i = 0; i < NPARAM; i++ ) {
        noveltyDB[i].push_back({});
    }
    otHH = OT;
    oteHH = TOTALT;
    stageHH = stNoveltySearch;
    calcBestHH = true;
    calcExceedHH = true;
    
    stims.clear();
    wave_pop_init( stims, GAPOP );

    // Stage: Novelty search
    for ( size_t generation = 0; generation < nGenerations && !*stopFlag; ++generation ) {
        cout << "Novelty search, generation " << generation << endl;
        reset(holdingVar, pertFac);
        size_t sn[GAPOP] = {};
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU(t);
            for (size_t i = 0; i < GAPOP; ++i) {
                if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                {
                    for (size_t j = 0; j < NPARAM + 1; ++j) {
                        float tmp = stims[i].V[sn[i]];
                        CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                    }
                    ++sn[i];
                }
            }
        }

        CHECK_CUDA_ERRORS( cudaMemcpy( exceedHH, d_exceedHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( nExceedHH, d_nExceedHH, NPOP*sizeof(int), cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( bestHH, d_bestHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( nBestHH, d_nBestHH, NPOP*sizeof(int), cudaMemcpyDeviceToHost ) );

        noveltyBundle bundle;
        double avgNovelty = 0;
        int numNew = 0;
        for ( size_t i = 0; i < GAPOP; ++i ) {
            stims[i].fit = 0.0;
            for ( size_t j = 1; j < NPARAM + 1; j++ ) {
                bundle.novelty[0] = exceedHH[i * (NPARAM + 1) + j];
                bundle.novelty[1] = nExceedHH[i * (NPARAM + 1) + j];
                bundle.novelty[2] = bestHH[i * (NPARAM + 1) + j];
                bundle.novelty[3] = nBestHH[i * (NPARAM + 1) + j];
                double least = 1e9;
                for ( noveltyBundle &p : noveltyDB[j-1] ) {
                    double dist = noveltyDistance(p, bundle);
                    if ( least > dist ) {
                        least = dist;
                        if ( least < noveltyThreshold )
                            break;
                    }
                }
                stims[i].fit += least / noveltyDB[j-1].size(); // Bias fitness, but not novelty, by count to boost params with fewer waves
                avgNovelty += least;
                if ( least > noveltyThreshold ) {
                    bundle.wave = stims[i];
                    noveltyDB[j-1].push_back(bundle);
                    ++numNew;
                }
            }
        }
        cout << "Average novelty value: " << (avgNovelty / GAPOP) << ", " << numNew << " new waves" << endl;

        procreatePop(stims);
    }
    
    for ( int i = 0; i < NPARAM; i++ ) {
        cout << "Parameter " << (i+1) << ": " << (noveltyDB[i].size()-1) << " touchstone waves" << endl;
        if ( noveltyDB[i].size() == 1 ) {
            cout << "Waveforms for this parameter will be generated from scratch. You may need to "
                    "decrease the novelty threshold or fit this parameter using a different method." << endl;
            noveltyDB[i].clear();
        } else {
            double sums[4] = {}, maxV[4] = {};
            for ( noveltyBundle &p : noveltyDB[i] ) {
                for ( int j = 0; j < 4; j++ ) {
                    sums[j] += p.novelty[j];
                    if ( maxV[j] < p.novelty[j] ) {
                        maxV[j] = p.novelty[j];
                    }
                }
            }
            cout << "\t\texceed\tnExceed\tbest\tnBest" << endl;
            cout << "\taverage:\t" << sums[0]/(noveltyDB[i].size()-1) << '\t' << sums[1]/(noveltyDB[i].size()-1)
                 << '\t' << sums[2]/(noveltyDB[i].size()-1) << '\t' << sums[3]/(noveltyDB[i].size()-1) << endl;
            cout << "\tmax (unc.):\t" << maxV[0] << '\t' << maxV[1] << '\t' << maxV[2] << '\t' << maxV[3] << endl;
        }
    }
}

void WavegenNS::optimiseAll(int nGenerations, ostream &wavefile, ostream &currentfile, bool *stopFlag)
{
    for ( int k = 0; k < NPARAM && !*stopFlag; k++ ) {
        optimise(k, nGenerations, stopFlag);

        // Since procreateInitialisedPop does not alter the first few waves, it is safe to assume that stims[0] is the fittest:
        validate(stims[0], k, currentfile);

        for ( int i = 0; i < NPARAM; i++ )
            wavefile << (i==k) << " ";
        for ( int i = 0; i < NPARAM; i++ )
            wavefile << sigmaAdjust[i] << " ";
        wavefile << stims[0] << endl;
    }
}

void WavegenNS::optimise(int param, int nGenerations, bool *stopFlag)
{
    stageHH = stWaveformOptimise;
    calcBestHH = true;
    calcExceedHH = false;

    unsigned int VSize = NPOP*theSize( model.ftype );

    vector<inputSpec> initial;
    initial.reserve(noveltyDB[param].size() * optimiseInitProportion);
    sort(noveltyDB[param].begin(), noveltyDB[param].end(), fittestNovelty);
    for ( int i = 0; i < noveltyDB[param].size() * optimiseInitProportion; i++ ) {
        initial.push_back(noveltyDB[param].at(i).wave);
    }

    stims.clear();
    wave_pop_init_from( stims, GAPOP, initial );

    // Optimise
    for (size_t generation = 0; generation < nGenerations && !*stopFlag; ++generation) {
        cout << "Optimising parameter " << (param+1) << ", generation " << generation << endl;
        reset(holdingVar, pertFac);
        size_t sn[GAPOP] = {};
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU(t);
            for (size_t i = 0; i < GAPOP; ++i) {
                if ((sn[i] < stims[i].N) && ((t - DT < stims[i].st[sn[i]]) && (t >= stims[i].st[sn[i]]) || (stims[i].st[sn[i]] == 0)))
                {
                    for (size_t j = 0; j < NPARAM + 1; ++j) {
                        float tmp = stims[i].V[sn[i]];
                        CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[i * (NPARAM + 1) + j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                    }
                    ++sn[i];
                }
            }
        }

        CHECK_CUDA_ERRORS( cudaMemcpy( bestHH, d_bestHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( nBestHH, d_nBestHH, VSize, cudaMemcpyDeviceToHost ) );
        for (size_t i = 0; i < GAPOP; ++i) {
            stims[i].fit = bestHH[i * (NPARAM + 1) + param + 1] * nBestHH[i * (NPARAM + 1) + param + 1];
        }
        procreateInitialisedPop( stims, initial );
        cout << stims[0] << endl;
    }
}

void WavegenNS::validate(inputSpec &stim, int param, ostream &currentfile)
{
    // Header
    int blocklen = SIM_TIME/DT;
    int offset = param*(blocklen+10) + 6; // 6 header lines, 4 footer lines, including blanks
    currentfile << "# Parameter " << param << " final waveform evoked currents" << endl;
    currentfile << "# Waveform: see below" << endl;
    currentfile << "#MATLAB dlmread(file, '\\t', [" << offset << "+headerLines 0 " << (blocklen+6) << "+headerLines, " << (NPARAM+2) << ")" << endl;
    currentfile << "#" << endl;
    currentfile << "Time\tTuned";
    for ( int j = 0; j < NPARAM; j++ ) {
        if ( j == param )
            currentfile << "\tTarget";
        else
            currentfile << "\tParam " << j;
    }
    currentfile << endl;

    // Run one epoch, writing current to errHH and recording observation window start/end
    stageHH = stObservationWindow;
    reset(holdingVar, pertFac);
    size_t sn = 0;
    scalar current[NPARAM + 1];
    unsigned int VSize = (NPARAM + 1) * theSize(model.ftype);
    int iT = 0;
    for ( double t = 0.0; t < SIM_TIME; t += DT, ++iT ) {
        stepTimeGPU(t);
        if ((sn < stim.N) && ((t - DT < stim.st[sn]) && (t >= stim.st[sn]) || (stim.st[sn] == 0)))
        {
            for (size_t j = 0; j < NPARAM + 1; ++j) {
                float tmp = stim.V[sn];
                CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
            }
            ++sn;
        }
        CHECK_CUDA_ERRORS( cudaMemcpy(current, d_errHH, VSize, cudaMemcpyDeviceToHost) );
        currentfile << t;
        for ( int j = 0; j < NPARAM + 1; j++ ) {
            currentfile << '\t' << current[j];
        }
        currentfile << endl;
    }
    CHECK_CUDA_ERRORS( cudaMemcpy( bestHH, d_bestHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( nBestHH, d_nBestHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( tStartHH, d_tStartHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( tEndHH, d_tEndHH, VSize, cudaMemcpyDeviceToHost ) );

    stim.fit = bestHH[param + 1] * nBestHH[param + 1];
    stim.ot = tStartHH[param + 1];
    stim.dur = tEndHH[param + 1] - stim.ot;

    currentfile << endl << "# Waveform for the above currents:" << endl;
    currentfile << "# " << stim << endl << endl;
}
