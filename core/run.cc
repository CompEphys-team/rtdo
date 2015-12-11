/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2015-12-08

--------------------------------------------------------------------------*/

#include <string>
#include <fstream>
#include <iostream>
#include <dlfcn.h>
#include "run.h"
#include "rt.h"
#include "globals.h"

static long thread=0;
static void *lib=0;
static int (*libmain)(const char*, const char*, const char*, const char*, int, int);
static int stop=0;
static daq_channel *active_in=0, *active_out=0;

void *vclaunch(void *);

void run_vclamp_start() {
    using std::endl;

    if ( lib ) {
        run_vclamp_stop();
    }

    stop = 0;
    rtdo_set_sampling_rate(sim_params.dt, 1);
    active_in =& daqchan_cin;
    active_out =& daqchan_vout;
    thread = rtdo_thread_create(vclaunch, 0, 1000000);
}

void run_vclamp_stop() {
    stop = 1;
    if ( thread ) {
        rtdo_thread_join(thread);
        rtdo_stop();
        thread = 0;
        active_in = 0;
        active_out = 0;
    }
    if ( lib ) {
        dlclose(lib);
        lib = 0;
    }
}

void *vclaunch(void *unused) {
    using std::endl;

    std::string cmd, instancedir = sim_params.modeldir + "/model.running";
    cmd = "rm -rf " + instancedir;
    system(cmd.c_str());

    cmd = "cp -r " + sim_params.modeldir + "/model " + instancedir;
    system(cmd.c_str());

    std::string fname = instancedir + "/HHVClampParameters.h";
    std::ofstream os( fname.c_str() );
    os << "#define NPOP " << sim_params.nPop << endl;
    os << "#define TOTALT " << 0 << endl;
    os << "#define fixGPU " << 0 << endl;
    os << "#define INoiseSTD " << 0 << endl;
    os << "#define DT " << sim_params.dt << endl;
    os << "#define RTDO" << endl;
    os.close();

    setenv("RTDO", "y", true);
    cmd = "cd " + instancedir + " && make clean && make release";
    if ( system(cmd.c_str()) )
        return (void *)EXIT_FAILURE;

    fname = instancedir + "/VClampGA.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        std::cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    if ( !(*(void**)(&libmain) = dlsym(lib, "vclamp")) ) {
        std::cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    libmain("live", sim_params.outdir.c_str(), sim_params.vc_wavefile.c_str(),
            sim_params.sigfile.c_str(), 1, -1);
    return 0;
}

daq_channel *run_get_active_outchan() {
    return active_out;
}

daq_channel *run_get_active_inchan() {
    return active_in;
}

void run_digest(double best_err, double mavg, int nextS) {
    std::cerr << best_err << " " << mavg << " " << nextS << std::endl;
}

int run_check_break() {
    return stop;
}

