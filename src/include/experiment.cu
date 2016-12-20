#ifndef EXPERIMENT_CU
#define EXPERIMENT_CU

#include "experiment_globals.h"
#include "cuda_helper.h" // For syntax highlighting only

void libManualInit(MetaModel &m) // Must be called separately (through Experiment_Global::init())
{
    allocateMem();
    initialize();
    Experiment_Global::populate(m);
}

void __attribute__ ((constructor)) libInit()
{
    Experiment_Global::init =& libManualInit;
    Experiment_Global::push =& pushHHStateToDevice;
    Experiment_Global::pull =& pullHHStateFromDevice;
    Experiment_Global::step =& stepTimeGPU;
    Experiment_Global::reset =& initialize;
    Experiment_Global::createSimulator =& Experiment_Global::createSim;
    Experiment_Global::destroySimulator =& Experiment_Global::destroySim;

    Experiment_Global::t =& t;
    Experiment_Global::iT =& iT;
}

void libExit()
{
    freeMem();
    cudaDeviceReset();
    Experiment_Global::init = 0;
    Experiment_Global::push = 0;
    Experiment_Global::pull = 0;
    Experiment_Global::step = 0;
    Experiment_Global::reset = 0;
    Experiment_Global::createSimulator = 0;
    Experiment_Global::destroySimulator = 0;

    Experiment_Global::t = 0;
    Experiment_Global::iT = 0;
}

#endif
