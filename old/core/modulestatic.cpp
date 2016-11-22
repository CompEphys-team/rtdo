/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-05-05

--------------------------------------------------------------------------*/
#include "experiment.h"
#include "wavegenNS.h"
#include <dlfcn.h>

void *Experiment::openLibrary()
{
    dlerror();
    void *lib;
    if ( ! (lib = dlopen(SOURCEDIR "/simulation/VClampGA.so", RTLD_NOW)) ) {
        throw runtime_error(dlerror());
    }
    return lib;
}

void Experiment::closeLibrary(void *lib)
{
    dlclose(lib);
}

Experiment *Experiment::create(void *lib)
{
    Experiment *exp;
    Experiment *(*vccreate)(conf::Config *cfg, int logSize, ostream &logOutput, size_t channel, size_t nchans);
    dlerror();
    if ( !(*(void**)(&vccreate) = dlsym(lib, "VClampCreate")) ) {
        string err(dlerror());
        dlclose(lib);
        throw runtime_error(err);
    }
    exp = vccreate(config, config->vc.popsize, cout, 0, 1);
    exp->initModel();
    return exp;
}

void Experiment::destroy(void *lib, Experiment **exp)
{
    void (*vcdestroy)(Experiment **);
    if ( !(*(void**)(&vcdestroy) = dlsym(lib, "VClampDestroy")) ) {
        throw runtime_error(dlerror());
    }
    vcdestroy(exp);
}


void *WavegenNSVirtual::openLibrary()
{
    dlerror();
    void *lib;
    if ( ! (lib = dlopen(SOURCEDIR "/wavegenNS/WaveGen.so", RTLD_NOW)) ) {
        throw runtime_error(dlerror());
    }
    return lib;
}

void WavegenNSVirtual::closeLibrary(void *lib)
{
    dlclose(lib);
}

WavegenNSVirtual *WavegenNSVirtual::create(void *lib)
{
    WavegenNSVirtual *exp;
    WavegenNSVirtual *(*wgcreate)(conf::Config *cfg);
    dlerror();
    if ( !(*(void**)(&wgcreate) = dlsym(lib, "WavegenCreate")) ) {
        string err(dlerror());
        dlclose(lib);
        throw runtime_error(err);
    }
    if ( config->model.obj->genn_float() ) {
        float (*simF)(float*, float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            string err(dlerror());
            dlclose(lib);
            throw runtime_error(err);
        }
        RealtimeEnvironment::env()->setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double*, double);
        if ( !(*(void**)(&simD) = dlsym(lib, "simulateSingleNeuron")) ) {
            string err(dlerror());
            dlclose(lib);
            throw runtime_error(err);
        }
        RealtimeEnvironment::env()->setSimulator(simD);
    }

    exp = wgcreate(config);
    return exp;
}

void WavegenNSVirtual::destroy(void *lib, WavegenNSVirtual **exp)
{
    void (*wgdestroy)(WavegenNSVirtual **);
    if ( !(*(void**)(&wgdestroy) = dlsym(lib, "WavegenDestroy")) ) {
        throw runtime_error(dlerror());
    }
    wgdestroy(exp);
}
