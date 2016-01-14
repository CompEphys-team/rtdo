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
#include <sstream>
#include <dlfcn.h>
#include <stdio.h>
#include "run.h"
#include "rt.h"
#include "globals.h"
#include "config.h"
#include "xmlmodel.h"

#define BRIDGE_START "Bridge code begin >>>"
#define BRIDGE_END "<<< Bridge code end"

using namespace std;

static long thread=0, sqthread=0;
static void *lib=0;
static int (*libmain)(const char*, const char*, const char*);
static int stop=0;
static daq_channel *active_in=0, *active_out=0;

static queue<double> samples_q;

void *vclaunch(void *);
void *sqread(void *);

string basename_nosuffix(const string& path) {
    int lastslash = path.find_last_of('/');
    int lastperiod = path.find_last_of('.');
    if ( lastslash && lastslash+1 < lastperiod ) {
        return path.substr(lastslash+1, lastperiod-lastslash-1);
    } else {
        return string();
    }
}

int compile_model() {
    string cxxflags = string("CXXFLAGS=\"$CXXFLAGS -std=c++11 -DDT=") + to_string(config.io.dt) + "\" ";
    string nvccflags = string("NVCCFLAGS=\"$NVCCFLAGS -DDT=") + to_string(config.io.dt) + "\" ";
    int ret=0;

    XMLModel model(config.model.deffile);
    string modelname = model.generateDefinition(XMLModel::VClamp, config.vc.popsize, INSTANCEDIR);

    string cmd = string("cd ") + INSTANCEDIR + " && " + cxxflags + "buildmodel.sh " + modelname + " 0 2>&1";
    cout << cmd << endl;
    FILE *is = popen(cmd.c_str(), "r");
    char buffer[1024];
    stringstream dump;
    while ( fgets(buffer, 1024, is) ) {
        dump << buffer;
        if ( strstr(buffer, "error") || strstr(buffer, "Error") )
            ret = 1;
    }
    pclose(is);
    cout << dump.str();
    if ( ret ) {
        cerr << "Model build failed." << endl;
        return 1;
    }

    string includeflags = string("INCLUDE_FLAGS='")
            + "-include " + INSTANCEDIR + "/" + modelname + "_CODE/runner.cc "
            + "-include " INSTANCEDIR + "/" + modelname + ".cc' ";
#ifdef _DEBUG
    cmd = string("cd ") + SIMDIR
            + " && " + "make clean -I " + INSTANCEDIR
            + " && " + cxxflags + nvccflags + includeflags + "make debug -I " + INSTANCEDIR;
#else
    cmd = string("cd ") + SIMDIR
            + " && " + "make clean -I " + INSTANCEDIR
            + " && " + cxxflags + nvccflags + includeflags + "make release -I " + INSTANCEDIR;
#endif
    cout << cmd << endl;
    if ( (ret = system(cmd.c_str())) ) {
        cerr << "Compilation failed." << endl;
        return 1;
    }

    cout << "Compilation successful." << endl;
    return 0;
}

void run_vclamp_start() {
    using std::endl;

    if ( lib ) {
        run_vclamp_stop();
    }

    while ( !samples_q.empty() )
        samples_q.pop();

    stop = 0;
    rtdo_set_sampling_rate(config.io.dt, 1);
    for ( vector<daq_channel *>::iterator it = config.io.channels.begin(); it != config.io.channels.end(); ++it ) {
        if ( *it == config.vc.in ) {
            active_in = *it;
            rtdo_set_channel_active(active_in->handle, 1);
        } else if ( *it == config.vc.out ) {
            active_out = *it;
            rtdo_set_channel_active(active_out->handle, 1);
        } else {
            rtdo_set_channel_active((*it)->handle, 0);
        }
    }
    if ( !active_in || !active_out )
        return;
    thread = rtdo_thread_create(vclaunch, 0, 1000000);
    sqthread = rtdo_thread_create(sqread, 0, 1000000);
}

void run_vclamp_stop() {
    stop = 1;
    if ( thread ) {
        rtdo_thread_join(thread);
        rtdo_thread_join(sqthread);
        rtdo_stop();
        thread = 0;
        sqthread = 0;
    }
    if ( lib ) {
        dlclose(lib);
        lib = 0;
    }
    rtdo_write_now(active_out->handle, active_out->offset);
    active_in = 0;
    active_out = 0;
}

void *vclaunch(void *unused) {
    string fname = string(SIMDIR) + "/VClampGA.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        std::cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    if ( !(*(void**)(&libmain) = dlsym(lib, "vclamp")) ) {
        std::cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    libmain("live", config.output.dir.c_str(), config.vc.wavefile.c_str());
    return 0;
}

void *sqread(void *) {
    std::string fname = config.output.dir + "/samples.tmp";
    std::ofstream os(fname.c_str());
    RTIME delay = nano2count(0.1e6);

    while( ! stop ) {
        rt_sleep(delay);
        rt_make_soft_real_time();
        while( ! samples_q.empty() ) {
            os << samples_q.front() << "\n";
            samples_q.pop();
        }
        os.flush();
    }

    os.close();
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

double run_getsample(float t) {
    daq_channel *in = run_get_active_inchan();
    int err=0;
    double sample = rtdo_get_data(in->handle, &err);
    samples_q.push(sample);
    return sample;
}
void run_setstimulus(inputSpec I) {
    daq_channel *out = run_get_active_outchan();
    int ret = rtdo_set_stimulus(out->handle, I.baseV, I.N, I.V.data(), I.st.data(), I.t);
    if ( ret )
        std::cerr << "Error " << ret << " setting stimulus waveform" << std::endl;
}

