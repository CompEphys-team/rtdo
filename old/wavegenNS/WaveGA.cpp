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
double VSTEP0 = -60.0;
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


#include "WaveGA.h"
#include "waveHelper.h"
#include <cuda.h>
#include <array>

#include "wavegenNS.h"

class WavegenNS : public WavegenNSVirtual
{
public:
    WavegenNS(conf::Config *cfg);
    ~WavegenNS() {}

    void runAll(std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag = nullptr);
    void adjustSigmas();
    void noveltySearch(bool *stopFlag = nullptr);
    void optimiseAll(std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag = nullptr);
    void optimise(int param, ostream &wavefile, ostream &currentfile, bool *stopFlag = nullptr);
    void validate(inputSpec &stim, int param, std::ostream &currentfile);
    inputSpec validate(inputSpec &stim, vector<vector<double>> &Isyns, vector<vector<double>> &modelCurrents, int param = 0);

private:
    vector<inputSpec> stims;
    scalar holdingVar[NVAR], singleParamIni[NPARAM];
    array<vector<noveltyBundle>, NPARAM> noveltyDB;

    NNmodel model;
};

extern "C" WavegenNSVirtual *WavegenCreate(conf::Config *cfg)
{
    return new WavegenNS(cfg);
}

extern "C" void WavegenDestroy(WavegenNSVirtual **_this)
{
    delete *_this;
    *_this = NULL;
}


WavegenNS::WavegenNS(conf::Config *cfg) :
    WavegenNSVirtual(cfg)
{
    // build the neuronal circuitery
    modelDefinition( model );
    allocateMem();
    initialize();
    rtdo_init_bridge();

    // If I ever start using more than one object at a time, this is going to have to go internal:
    VSTEP0 = cfg->model.obj->baseV();

    clampGainHH = cfg->vc.gain;
    accessResistanceHH = cfg->vc.resistance;

    timeToleranceHH = cfg->wg.tolTime;
    currentToleranceHH = cfg->wg.tolCurrent;
    deltaToleranceHH = cfg->wg.tolDelta;

    for ( int i = 0; i < NPARAM; i++ ) {
        sigmaAdjust[i] = 1;
    }

    // Get steady-state variable values at holding potential
    for ( int i = 0; i < NVAR; i++ )
        holdingVar[i] = mvar[i][0];
    for ( int i = 0; i < NPARAM; i++ )
        singleParamIni[i] = mparam[i][0];
    scalar currents[NCURRENTS];
    for ( double t = 0.0; t < 10000.0; t += DT ) {
        simulateSingleNeuron(holdingVar, singleParamIni, currents, VSTEP0);
    }
}

void WavegenNS::runAll(ostream &wavefile, ostream &currentfile, bool *stopFlag)
{
    if ( !stopFlag )
        stopFlag =& this->stopFlag;

    adjustSigmas();
    noveltySearch(stopFlag);
    optimiseAll(wavefile, currentfile, stopFlag);
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

    otHH = OT;
    oteHH = TOTALT;
    stageHH = stDetuneAdjust;
    
    stims.clear();
    wave_pop_init( stims, GAPOP );

    // Stage: Adjust sigmas to detune with similar results:
    *log << "Adjusting parameter sigmas..." << endl;
    for ( int r = 0; r < 2; r++ ) {
        reset(holdingVar, pertFac);
        size_t sn[GAPOP] = {};
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU();
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
        *log << cfg->model.obj->adjustableParams().at(i).name << " sigma adjustment: " << sigmaAdjust[i] << endl;
    }
    *log << endl;
}

void WavegenNS::noveltySearch(bool *stopFlag)
{
    if ( !stopFlag )
        stopFlag =& this->stopFlag;

    unsigned int VSize = NPOP*theSize( model.ftype );
    for ( int i = 0; i < NPARAM; i++ ) {
        noveltyDB[i].push_back(noveltyBundle());
    }
    otHH = OT;
    oteHH = TOTALT;
    stageHH = stNoveltySearch;
    calcBestHH = true;
    calcExceedHH = true;
    calcSeparationHH = true;
    
    stims.clear();
    wave_pop_init( stims, GAPOP );

    double nSteps = (TOTALT-OT)/DT;

    // Stage: Novelty search
    for ( size_t generation = 0; generation < cfg->wg.ngen && !*stopFlag; ++generation ) {
        *log << "Novelty search, generation " << generation << endl;
        reset(holdingVar, pertFac);
        size_t sn[GAPOP] = {};
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU();
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
        CHECK_CUDA_ERRORS( cudaMemcpy( separationHH, d_separationHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( nSeparationHH, d_nSeparationHH, NPOP*sizeof(int), cudaMemcpyDeviceToHost ) );

        double avgNovelty = 0;
        int numNew = 0;
        for ( size_t i = 0; i < GAPOP; ++i ) {
            stims[i].fit = 0.0;
            for ( size_t j = 1; j < NPARAM + 1; j++ ) {
                noveltyBundle bundle(i * (NPARAM + 1) + j);
                double least = 1e9;
                for ( noveltyBundle &p : noveltyDB[j-1] ) {
                    double dist = noveltyDistance(p, bundle) / nSteps; // Normalise all novelty dimensions
                    if ( least > dist ) {
                        least = dist;
                        if ( least < cfg->wg.ns_noveltyThreshold )
                            break;
                    }
                }
                stims[i].fit += least / noveltyDB[j-1].size(); // Bias fitness, but not novelty, by count to boost params with fewer waves
                avgNovelty += least;
                if ( least > cfg->wg.ns_noveltyThreshold ) {
                    bundle.wave = stims[i];
                    noveltyDB[j-1].push_back(bundle);
                    ++numNew;
                }
            }
        }
        *log << "Average novelty value: " << (avgNovelty / GAPOP) << ", " << numNew << " new waves" << endl;

        procreatePop(stims);
    }
    
    for ( int i = 0; i < NPARAM; i++ ) {
        noveltyDB[i].erase(noveltyDB[i].begin()); // Remove initial dummy
        *log << cfg->model.obj->adjustableParams().at(i).name << ": " << noveltyDB[i].size() << " reference waves" << endl;
        if ( !noveltyDB[i].size() ) {
            *log << "Waveforms for this parameter will be generated from scratch during optimisation. "
                 << "You may need to decrease the novelty threshold or fit this parameter using a different method." << endl;
        } else {
            double sums[NNOVELTY+1] = {}, maxV[NNOVELTY+1] = {};
            for ( noveltyBundle &p : noveltyDB[i] ) {
                for ( int j = 0; j < NNOVELTY; j++ ) {
                    sums[j] += p.novelty[j];
                    if ( maxV[j] < p.novelty[j] ) {
                        maxV[j] = p.novelty[j];
                    }
                }
                double f = p.fitness();
                sums[NNOVELTY] += f;
                if ( maxV[NNOVELTY] < f )
                    maxV[NNOVELTY] = f;
            }
            *log << "\t\texceed\tnExceed\tbest\tnBest\tseparation\tnSep\tfitness" << endl;
            *log << "\tmean:";
            for ( int j = 0; j < NNOVELTY+1; j++ ) {
                *log << '\t' << sums[j]/noveltyDB[i].size();
            }
            *log << endl;
            *log << "\tmax (unc.):";
            for ( int j = 0; j < NNOVELTY+1; j++ ) {
                *log << '\t' << maxV[j];
            }
            *log << endl;
            if ( maxV[NNOVELTY] == 0 ) {
                *log << "Waveforms for this parameter will be generated from scratch during optimisation. "
                     << "You may need to decrease the novelty threshold or fit this parameter using a different method." << endl;
            }
        }
    }
}

void WavegenNS::optimiseAll(ostream &wavefile, ostream &currentfile, bool *stopFlag)
{
    if ( !stopFlag )
        stopFlag =& this->stopFlag;

    for ( int k = 0; k < NPARAM && !*stopFlag; k++ ) {
        optimise(k, wavefile, currentfile, stopFlag);
    }
}

void WavegenNS::optimise(int param, ostream &wavefile, ostream &currentfile, bool *stopFlag)
{
    if ( !stopFlag )
        stopFlag =& this->stopFlag;

    stageHH = stWaveformOptimise;
    calcBestHH = false;
    calcExceedHH = false;
    calcSeparationHH = true;
    otHH = OT;
    oteHH = TOTALT;

    unsigned int VSize = NPOP*theSize( model.ftype );

    vector<inputSpec> initial;
    if ( noveltyDB[param].size() ) {
        initial.reserve(noveltyDB[param].size() * cfg->wg.ns_optimiseProportion);
        sort(noveltyDB[param].begin(), noveltyDB[param].end(), fittestNovelty);
        for ( int i = 0; i < noveltyDB[param].size() * cfg->wg.ns_optimiseProportion && noveltyDB[param].at(i).fitness() > 0; i++ ) {
            initial.push_back(noveltyDB[param].at(i).wave);
        }
    }

    stims.clear();
    wave_pop_init_from( stims, GAPOP, initial );

    // Optimise
    for (size_t generation = 0; generation < cfg->wg.ns_ngenOptimise && !*stopFlag; ++generation) {
        *log << "Optimising parameter " << cfg->model.obj->adjustableParams().at(param).name << ", generation " << generation << endl;
        reset(holdingVar, pertFac);
        size_t sn[GAPOP] = {};
        for (double t = 0.0; t < SIM_TIME; t += DT) {
            stepTimeGPU();
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

        CHECK_CUDA_ERRORS( cudaMemcpy( separationHH, d_separationHH, VSize, cudaMemcpyDeviceToHost ) );
        CHECK_CUDA_ERRORS( cudaMemcpy( nSeparationHH, d_nSeparationHH, VSize, cudaMemcpyDeviceToHost ) );
        for (size_t i = 0; i < GAPOP; ++i) {
            noveltyBundle tmp(i * (NPARAM + 1) + param + 1);
            stims[i].fit = tmp.fitness();
        }
        procreateInitialisedPop( stims, initial );
        *log << stims[0] << endl;
    }

    // Since procreateInitialisedPop does not alter the first few waves, it is safe to assume that stims[0] is the fittest:
    validate(stims[0], param, currentfile);

    wavefile << "# " << cfg->model.obj->adjustableParams().at(param).name << endl;
    for ( int i = 0; i < NPARAM; i++ )
        wavefile << (i==param) << " ";
    for ( int i = 0; i < NPARAM; i++ )
        wavefile << sigmaAdjust[i] << " ";
    wavefile << stims[0] << endl;
}

void WavegenNS::validate(inputSpec &stim, int param, ostream &currentfile)
{
    // Header
    int blocklen = SIM_TIME/DT;
    int headerlen = 8;
    int footerlen = 9;
    int offset = param*(blocklen+headerlen+footerlen) + headerlen;
    int offsetBottom = offset + blocklen + headerlen;
    currentfile << "# Parameter " << param << " final waveform evoked currents" << endl;
    currentfile << "# Waveform: see below" << endl;
    currentfile << "#MATLAB time_p" << param << " = dlmread(file, '\\t', "
                << "[" << offset << "+headerLines 0 " << offsetBottom << "+headerLines 0]);" << endl;
    currentfile << "#MATLAB waveform_p" << param << " = dlmread(file, '\\t', "
                << "[" << offset << "+headerLines 1 " << offsetBottom << "+headerLines 1]);" << endl;
    currentfile << "#MATLAB reference_p" << param << " = dlmread(file, '\\t', "
                << "[" << offset << "+headerLines 2 " << offsetBottom << "+headerLines 2]);" << endl;
    currentfile << "#MATLAB currents_p" << param << " = dlmread(file, '\\t', "
                << "[" << offset << "+headerLines 3 " << offsetBottom << "+headerLines " << (NPARAM+2) << "]);" << endl;
    currentfile << "#" << endl;
    currentfile << "Time\tVoltage\tReference";
    for ( int j = 0; j < NPARAM; j++ ) {
        currentfile << '\t' << cfg->model.obj->adjustableParams().at(j).name;
    }
    for ( int j = 0; j < NCURRENTS; j++ ) {
        currentfile << '\t' << cfg->model.obj->currents().at(j).name;
    }
    currentfile << endl;

    // Run one epoch, writing current to $(err) and recording observation window start/end
    stageHH = stObservationWindowSeparation;
    reset(holdingVar, pertFac);
    size_t sn = 0;
    scalar current[NPARAM + 1];
    unsigned int VSize = (NPARAM + 1) * theSize(model.ftype);
    int iT = 0;
    for ( double t = 0.0; t < SIM_TIME; t += DT, ++iT ) {
        stepTimeGPU();
        if ((sn < stim.N) && ((t - DT < stim.st[sn]) && (t >= stim.st[sn]) || (stim.st[sn] == 0)))
        {
            for (size_t j = 0; j < NPARAM + 1; ++j) {
                float tmp = stim.V[sn];
                CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[j], &tmp, sizeof( float ), cudaMemcpyHostToDevice ) );
                stepVGHH[0] = tmp;
            }
            ++sn;
        }
        CHECK_CUDA_ERRORS( cudaMemcpy(current, d_errHH, VSize, cudaMemcpyDeviceToHost) );
        currentfile << t << '\t' << stepVGHH[0];
        for ( int j = 0; j < NPARAM + 1; j++ ) {
            currentfile << '\t' << current[j];
        }
        for ( int j = 0; j < NCURRENTS; j++ ) {
            CHECK_CUDA_ERRORS( cudaMemcpy(mcurrents[j], d_mcurrents[j], theSize(model.ftype), cudaMemcpyDeviceToHost) );
            currentfile << '\t' << mcurrents[j][0];
        }
        currentfile << endl;
    }

    CHECK_CUDA_ERRORS( cudaMemcpy( separationHH, d_separationHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( nSeparationHH, d_nSeparationHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( tStartHH, d_tStartHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( tEndHH, d_tEndHH, VSize, cudaMemcpyDeviceToHost ) );

    noveltyBundle tmp(param + 1);
    stim.fit = tmp.fitness();
    stim.ot = tStartHH[param + 1];
    stim.dur = tEndHH[param + 1] - stim.ot;

    currentfile << endl << "# Waveform for the above currents:" << endl;
    currentfile << "# " << stim << endl << endl;
    currentfile << "# Observation window for the above:" << endl;
    currentfile << "start\tend" << endl;
    currentfile << stim.ot << '\t' << (stim.ot+stim.dur) << endl << endl << endl;
}

inputSpec WavegenNS::validate(inputSpec &stim, vector<vector<double>> &Isyns, vector<vector<double>> &modelCurrents, int param)
{
    Isyns.clear();
    modelCurrents.clear();
    Isyns.resize(NPARAM+1, vector<double>(SIM_TIME/DT, 0.0));
    modelCurrents.resize(NCURRENTS, vector<double>(SIM_TIME/DT, 0.0));

    stageHH = stObservationWindowSeparation;
    reset(holdingVar, pertFac);
    noveltyBundle prevNB{}, cumNB{};
    size_t sn = 0;
    unsigned int VSize = (NPARAM + 1) * theSize(model.ftype);
    int iT = 0;
    for ( double t = 0.0; t < SIM_TIME; t += DT, ++iT ) {
        stepTimeGPU();
        if ((sn < stim.N) && ((t - DT < stim.st[sn]) && (t >= stim.st[sn]) || (stim.st[sn] == 0)))
        {
            for (size_t j = 0; j < NPARAM + 1; ++j) {
                scalar tmp = stim.V[sn];
                CHECK_CUDA_ERRORS( cudaMemcpy( &d_stepVGHH[j], &tmp, theSize(model.ftype), cudaMemcpyHostToDevice ) );
            }
            ++sn;
        }
        CHECK_CUDA_ERRORS( cudaMemcpy(errHH, d_errHH, VSize, cudaMemcpyDeviceToHost) );
        for ( int j = 0; j < NPARAM + 1; j++ ) {
            Isyns[j][iT] = errHH[j];
        }
        for ( int j = 0; j < NCURRENTS; j++ ) {
            CHECK_CUDA_ERRORS( cudaMemcpy(mcurrents[j], d_mcurrents[j], theSize(model.ftype), cudaMemcpyDeviceToHost) );
            modelCurrents[j][iT] = mcurrents[j][0];
        }

        if ( t+DT > stim.ot && t < stim.ot + stim.dur ) {
            // Grab only the fitness-relevant values
            CHECK_CUDA_ERRORS( cudaMemcpy( separationHH, d_separationHH, VSize, cudaMemcpyDeviceToHost ) );
            CHECK_CUDA_ERRORS( cudaMemcpy( nSeparationHH, d_nSeparationHH, VSize, cudaMemcpyDeviceToHost ) );
            noveltyBundle tmp(param + 1);
            if ( t > stim.ot )
                for ( int j = 0; j < NNOVELTY; j++ )
                    if ( tmp.novelty[j] > prevNB.novelty[j] )
                        cumNB.novelty[j] += tmp.novelty[j] - prevNB.novelty[j];
            prevNB = tmp;
        }
    }

    CHECK_CUDA_ERRORS( cudaMemcpy( separationHH, d_separationHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( nSeparationHH, d_nSeparationHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( tStartHH, d_tStartHH, VSize, cudaMemcpyDeviceToHost ) );
    CHECK_CUDA_ERRORS( cudaMemcpy( tEndHH, d_tEndHH, VSize, cudaMemcpyDeviceToHost ) );

    noveltyBundle tmp(param + 1);
    inputSpec ret(stim);
    ret.fit = tmp.fitness();
    ret.ot = tStartHH[param + 1];
    ret.dur = tEndHH[param + 1] - ret.ot;

    stim.fit = cumNB.fitness();

    return ret;
}
