/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-05-05

--------------------------------------------------------------------------*/
#include "experiment.h"
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

