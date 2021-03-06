/*--------------------------------------------------------------------------
Author: Thomas Nowotny

Institute: Informatics
University of Sussex
Brighton BN1 9QJ, UK

email to:  t.nowotny@sussex.ac.uk

initial version: 2014-06-26

--------------------------------------------------------------------------*/

//--------------------------------------------------------------------------
/*! \file VClampGA.cu

\brief Main entry point for the GeNN project demonstrating realtime fitting of a neuron with a GA running mostly on the GPU.
*/
//--------------------------------------------------------------------------
#define _V_CLAMP
#ifdef _V_CLAMP
#include "VClampGA.h"
#include "realtimeenvironment.h"
#include "experiment.h"
#include <numeric>

static NNmodel model;

class VClamp : public Experiment
{
public:
    VClamp(conf::Config *cfg, int logSize, ostream &logOutput, size_t channel, size_t nchans);
    ~VClamp() {}

    void initModel();
    void run(int nEpochs = 0);
    void cycle(bool fit);

    vector<vector<double>> stimulateModel(int idx);
    void injectModel(vector<double> params, int idx);
    void fixParameter(int paramIdx, double value);

    vector<double> getCCVoltageTrace(inputSpec I, int idx);

    void setLog(std::ostream *out, std::string closingMessage = string());

private:
    void runStim();
};

extern "C" Experiment *VClampCreate(conf::Config *cfg, int logSize, ostream &logOutput, size_t channel, size_t nchans)
{
    return new VClamp(cfg, logSize, logOutput, channel, nchans);
}

extern "C" void VClampDestroy(Experiment **_this)
{
    delete *_this;
    *_this = nullptr;
}


VClamp::VClamp(conf::Config *cfg, int logSize, ostream &logOutput, size_t channel, size_t nchans) :
    Experiment(cfg, channel)
{
    ifstream is(cfg->vc.wavefile);
    load_stim(is, pperturb, sigadjust, stims);
    is.close();
    for (int i = 0, k = pperturb.size(); i < k; i++) {
        for (int j = 0, l = pperturb[i].size(); j < l; j++) {
            cout << pperturb[i][j] << " ";
        }
        for (int j = 0, l = pperturb[i].size(); j < l; j++) {
            cout << sigadjust[i][j] << " ";
        }
        cout << endl;
        cout << stims[i] << endl;
        if ( stims[i].baseV != cfg->model.obj->baseV() )
            cerr << "Warning: Base voltage mismatch between model (" << cfg->model.obj->baseV() << " mV) and waveform ("
                 << stims[i].baseV << " mV)." << endl;
    }

    int Nstim = stims.size();
    errbuf = vector<vector<double>>(Nstim, vector<double>(MAVGBUFSZ));
    mavg = vector<double>(Nstim, 0);
    epos = vector<int>(Nstim, 0);
    initial = vector<int>(Nstim, 1);

    logger = make_shared<backlog::Backlog>(logSize, Nstim, &logOutput);
    if ( nchans > 1 )
        _data = make_shared<MultiChannelData>(Nstim, nchans);
    else
        _data = make_shared<SingleChannelData>(Nstim);
}

void VClamp::initModel()
{
    currentExperiment = this;
    *(logger->out) << "# VClamp initialising" << endl;
    if ( !model.final ) {
        modelDefinition( model );
        allocateMem();
        initialize();
        rtdo_init_bridge();
    }
    t = 0.0;
    epoch = 0;
    nextS = 0;
    models.clear();
    models.reserve(NPOP);
    for ( int i = 0; i < NPOP; i++ ) {
        models.emplace_back(i);
    }
    copyStateToDevice();
}

void VClamp::run(int nEpochs)
{
    currentExperiment = this;
    logger->printHeader("# VClamp running");
    if ( RealtimeEnvironment::env()->isSimulating() ) {
        scalar simulatorVars[NVAR], simulatorParams[NPARAM];
        for ( int i = 0; i < NVAR; i++ )
            simulatorVars[i] = variableIni[i];
        for ( int i = 0; i < NPARAM; i++ )
            simulatorParams[i] = aParamIni[i];
        RealtimeEnvironment::env()->setSimulatorVariables(simulatorVars);
        RealtimeEnvironment::env()->setSimulatorParameters(simulatorParams);
    }

	unsigned int VSize = NPOP*theSize( model.ftype );

    int epEnd = epoch + nEpochs;
    while ( !stopFlag && (!nEpochs || epoch < epEnd) )
    {
        runStim();
        CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );
        procreateGeneric();
        ++epoch;
    }
}

void VClamp::runStim()
{
    inputSpec I = stims[nextS];
    double lt = 0.0;
    int sn = 0;

    VCHH = true;
    stepVGHH = I.baseV;
    otHH = t + I.ot;
    oteHH = t + I.ot + I.dur;
    clampGainHH = cfg->vc.gain;
    accessResistanceHH = cfg->vc.resistance;
    simCyclesHH = cfg->model.cycles;
    truevar_init();

    _data->startSample(I, nextS);

    for (int iT = 0; iT < (I.t / DT); iT++) {
        double oldt = lt;
        IsynGHH = _data->nextSample(channel);
        stepTimeGPU();
        lt += DT;
        if ((sn < I.N) && ((oldt < I.st[sn]) && (lt >= I.st[sn]) || (I.st[sn] == 0))) {
            stepVGHH = I.V[sn];
            sn++;
        }
    }

    if ( RealtimeEnvironment::env()->idleTimeReporting() )
        cudaDeviceSynchronize();

    _data->endSample();
}

void VClamp::cycle(bool fit)
{
    currentExperiment = this;
    logger->printHeader("# VClamp cycling");
    errTupel errs[fit ? 1 : NPOP];
    unsigned int VSize = NPOP*theSize( model.ftype );
    for ( size_t i = 0; i < stims.size() && !stopFlag; i++ ) {
        nextS = i;
        runStim();
        CHECK_CUDA_ERRORS( cudaMemcpy( errHH, d_errHH, VSize, cudaMemcpyDeviceToHost ) );
        if ( fit ) {
            procreateGeneric();
        } else {
            // Update log without fitting, using all models
            logger->wait();
            for (int i = 0; i < NPOP; i++) {
                errs[i].id = i;
                errs[i].err = errHH[i];
            }
            qsort( (void *)errs, NPOP, sizeof( errTupel ), compareErrTupel );
            logger->touch(&errs[0], &errs[NPOP-1], epoch, nextS);
        }
        ++epoch;
    }

    if ( !fit )
        logger->wait();
}

vector<vector<double>> VClamp::stimulateModel(int idx)
{
    // As this is a local simulation only, do not update globals such as t, currentExperiment, or VClamp::nextS
    *(logger->out) << "# VClamp stimulating model index " << idx << endl;
    vector<vector<double>> traces;
    for ( size_t i = 0; i < stims.size() && !stopFlag; i++ ) {
        inputSpec I = stims[i];
        double lt = -I.t;
        int sn = 0;
        int tmax = I.t / DT;
        vector<double> trace(tmax);

        VCHH = true;
        stepVGHH = I.baseV;
        clampGainHH = cfg->vc.gain;
        accessResistanceHH = cfg->vc.resistance;

        scalar simulatorVars[NVAR], simulatorParams[NPARAM], currents[NCURRENTS];
        for ( int i = 0; i < NVAR; i++ )
            simulatorVars[i] = mvar[i][idx];
        for ( int i = 0; i < NPARAM; i++ )
            simulatorParams[i] = mparam[i][idx];

        // Attempt to foil instabilities
        if ( std::isnan(simulateSingleNeuron(simulatorVars, simulatorParams, currents, stepVGHH)) ) {
            for ( int i = 0; i < NVAR; i++ ) {
                simulatorVars[i] = 0.0;
            }
        }

        for (int iT = -tmax; iT < tmax; iT++) { // Start at negative t to achieve steady state by lt=0
            double oldt = lt;
            IsynGHH = simulateSingleNeuron(simulatorVars, simulatorParams, currents, stepVGHH);
            lt += DT;
            if ((sn < I.N) && ((oldt < I.st[sn]) && (lt >= I.st[sn]) || (I.st[sn] == 0))) {
                stepVGHH = I.V[sn];
                sn++;
            }
            if ( iT >= 0 ) {
                trace[iT] = IsynGHH;
            }
        }

        traces.push_back(trace);
    }

    return traces;
}

void VClamp::injectModel(std::vector<double> params, int idx)
{
    int i = 0;
    for ( double &p : params )
        mparam[i++][idx] = p;
}

void VClamp::fixParameter(int paramIdx, double value)
{
    for ( int i = 0; i < NPOP; i++ ) {
        mparam[paramIdx][i] = value;
    }
    for ( auto &p : pperturb ) {
        p.at(paramIdx) = 0;
    }
}

vector<double> VClamp::getCCVoltageTrace(inputSpec I, int idx)
{
    double lt = -2000;
    int sn = 0;
    int tmax = I.t / DT;
    vector<double> trace(tmax);

    int i = 0, voltIdx = 0;
    for ( auto v : config->model.obj->vars() ) {
        if ( !v.name.compare("V") )
            voltIdx = i;
        ++i;
    }

    VCHH = false;
    IsynGHH = I.baseV;

    scalar simulatorVars[NVAR], simulatorParams[NPARAM], currents[NCURRENTS];
    for ( int i = 0; i < NVAR; i++ )
        simulatorVars[i] = mvar[i][idx];
    for ( int i = 0; i < NPARAM; i++ )
        simulatorParams[i] = mparam[i][idx];

    for (int iT = lt/DT; iT < tmax; iT++) {
        double oldt = lt;
        simulateSingleNeuron(simulatorVars, simulatorParams, currents, stepVGHH);
        lt += DT;
        if ((sn < I.N) && ((oldt < I.st[sn]) && (lt >= I.st[sn]) || (I.st[sn] == 0))) {
            IsynGHH = I.V[sn];
            sn++;
        }
        if ( iT >= 0 ) {
            trace[iT] = simulatorVars[voltIdx];
        }
    }

    return trace;
}

void VClamp::setLog(std::ostream *out, std::string closingMessage)
{
    logger->wait();
    *logger->out << closingMessage;
    logger->out->flush();
    logger->out = out;
}


void Experiment::procreateGeneric()
{
    logger->wait();

    double amplitude = 0.1;
    double tmavg, delErr;

    static errTupel errs[NPOP];
    static int limiter = 0;

    for (int i = 0; i < NPOP; i++) {
        errs[i].id = i;
        errs[i].err = errHH[i];
    }
    qsort( (void *)errs, NPOP, sizeof( errTupel ), compareErrTupel );

    int k = NPOP / 4;
    logger->touch(&errs[0], &errs[k-1], epoch, nextS);

    // update moving averages
    epos[nextS] = (epos[nextS] + 1) % MAVGBUFSZ;
    if (initial[nextS]) {
        if (epos[nextS] == 0) {
            initial[nextS] = 0;
        }
        delErr = 2 * errs[0].err;
    }
    else {
        delErr = errbuf[nextS][epos[nextS]];
        mavg[nextS] -= delErr;
    }
    errbuf[nextS][epos[nextS]] = errs[0].err;
    mavg[nextS] += errbuf[nextS][epos[nextS]];
    tmavg = mavg[nextS] / MAVGBUFSZ;

    bool nextSChanged = false;
    if (errs[0].err < tmavg*0.8) {
        // we are getting better on this one -> adjust a different parameter combination
        nextS = (nextS + 1) % stims.size();
        nextSChanged = true;
    }
    if ( delErr > 1.01 * errs[0].err ) {
        limiter = (limiter < 3 ? 0 : limiter-3);
    }
    if ( ++limiter > 5 ) {
        // Stuck, move on
        nextS = (nextS + 1) % stims.size();
        limiter = 0;
        nextSChanged = true;
    }

    if ( nextSChanged ) {
        for ( int i = 0; i < NPOP; i++ ) {
            models[i].errDiff = 0;
        }

        // Skip waveforms with no fitting component
        while ( std::accumulate(pperturb.at(nextS).begin(), pperturb.at(nextS).end(), 0.0) == 0.0 )
            nextS = (nextS + 1) % stims.size();
    } else {
        for ( int i = 0; i < NPOP; i++ ) {
            models[i].diff();
        }
    }

    // First quarter: Elitism, no action
    // Second quarter: mutated elite without momentum
    for (int i = k; i < 2 * k; i++) {
        models[errs[i].id].copy(errs[i-k].id);
        models[errs[i].id].mutate(nextS, amplitude, false);
    }
    // Third quarter: mutated elite with momentum
    for (int i = 2 * k; i < 3 * k; i++) {
        models[errs[i].id].copy(errs[i - 2*k].id);
        models[errs[i].id].mutate(nextS, amplitude, true);
    }
    // Fourth quarter: Elite with random reset in the current parameter
    for (int i = 3 * k; i < NPOP; i++) {
        models[errs[i].id].copy(errs[i - 3*k].id);
        models[errs[i].id].reinit(nextS);
    }
    // Never ever: Add completely new models to a serial multiparameter fitting run
}

#endif
