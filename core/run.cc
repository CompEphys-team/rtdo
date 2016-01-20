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

using namespace std;

static long thread=0, sqthread=0;
static void *lib=0;
static int (*libmain)(const char*, const char*, const char*);
static inputSpec (*wgmain)(int, int);
static int stop=0;
static daq_channel *active_in=0, *active_out=0;

static queue<double> samples_q;

void *vclaunch(void *);
void *wglaunch(void *);
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

ostream &operator<<(ostream &os, inputSpec &I) {
    os << I.t << " " << I.ot << " " << I.dur << " " << I.baseV << " " << I.N << " ";
    for (int i = 0; i < I.N; i++) {
        os << I.st[i] << " ";
        os << I.V[i] << " ";
    }
    os << I.fit;
    return os;
}

istream &operator>>(istream &is, inputSpec &I) {
    double tmp;
    I.st.clear();
    I.V.clear();
    is >> I.t >> I.ot >> I.dur >> I.baseV >> I.N;
    for ( int i = 0; i < I.N; i++ ) {
        is >> tmp;
        I.st.push_back(tmp);
        is >> tmp;
        I.V.push_back(tmp);
    }
    is >> I.fit;
    return is;
}

int compile_model(XMLModel::outputType type) {
    string cxxflags = string("CXXFLAGS=\"$CXXFLAGS -std=c++11 -DDT=") + to_string(config->io.dt) + "\" ";
    string nvccflags = string("NVCCFLAGS=\"$NVCCFLAGS -DDT=") + to_string(config->io.dt) + "\" ";
    int ret=0;

    int popsize=0;
    string simulator(SOURCEDIR);
    switch ( type ) {
    case XMLModel::VClamp:
        popsize = config->vc.popsize;
        simulator += "/simulation";
        break;
    case XMLModel::WaveGen:
        popsize = config->wg.popsize;
        simulator += "/wavegen";
        break;
    }

    if ( !config->model.load() ) {
        cerr << "Error: Unable to load model file '" << config->model.deffile << "'." << endl;
        return 1;
    }

    string modelname = config->model.obj->generateDefinition(type, popsize, INSTANCEDIR);

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

    // Build single-neuron version
    string singlename = config->model.obj->generateDefinition(type, 1, INSTANCEDIR, true);
    cmd = string("cd ") + INSTANCEDIR + " && " + cxxflags + "buildmodel.sh " + singlename + " 0 2>&1";
    cout << cmd << endl;
    ret = system(cmd.c_str()); // Return value from buildmodel.sh is always 0, ignore it
    ofstream inc(string(INSTANCEDIR) + "/" + singlename + "_CODE/single.h");
    inc << "#define calcNeuronsCPU calcSingleNeuronCPU" << endl;
    inc << "#include \"neuronFnct.cc\"" << endl;
    inc << "#undef calcNeuronsCPU" << endl;
    inc.close();

    // Compile it all
    string includeflags = string("INCLUDE_FLAGS='")
            + "-include " + INSTANCEDIR + "/" + modelname + "_CODE/runner.cc "
            + "-include " + INSTANCEDIR + "/" + modelname + ".cc "
            + "-include " + INSTANCEDIR + "/" + singlename + "_CODE/single.h' ";
#ifdef _DEBUG
    cmd = string("cd ") + simulator
            + " && " + "make clean -I " + INSTANCEDIR
            + " && " + cxxflags + nvccflags + includeflags + "make debug -I " + INSTANCEDIR;
#else
    cmd = string("cd ") + simulator
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

bool run_vclamp_start() {
    if ( thread )
        return false;

    while ( !samples_q.empty() )
        samples_q.pop();

    stop = 0;
    rtdo_set_sampling_rate(config->io.dt, 1);
    for ( vector<daq_channel *>::iterator it = config->io.channels.begin(); it != config->io.channels.end(); ++it ) {
        if ( *it == config->vc.in ) {
            active_in = *it;
            if ( active_in->read_offset_later && active_in->read_offset_src ) {
                int err;
                double tmp = rtdo_read_now(active_in->read_offset_src->handle, &err);
                if ( err ) {
                    cerr << "Warning: Read offset for channel " << active_in->name << " failed with error code " << err << endl;
                } else {
                    active_in->offset = tmp;
                }
            }
            rtdo_set_channel_active(active_in->handle, 1);
        } else if ( *it == config->vc.out ) {
            active_out = *it;
            if ( active_out->read_offset_later && active_out->read_offset_src ) {
                int err;
                double tmp = rtdo_read_now(active_out->read_offset_src->handle, &err);
                if ( err ) {
                    cerr << "Warning: Read offset for channel " << active_out->name << " failed with error code " << err << endl;
                } else {
                    active_out->offset = tmp;
                }
            }
            rtdo_set_channel_active(active_out->handle, 1);
        } else {
            rtdo_set_channel_active((*it)->handle, 0);
        }
    }
    if ( !active_in || !active_out )
        return false;
    thread = rtdo_thread_create(vclaunch, 0, 1000000);
    sqthread = rtdo_thread_create(sqread, 0, 1000000);
    return true;
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
    string fname = string(SOURCEDIR) + "/simulation/VClampGA.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        std::cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    if ( !(*(void**)(&libmain) = dlsym(lib, "vclamp")) ) {
        std::cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    libmain("live", config->output.dir.c_str(), config->vc.wavefile.c_str());
    return 0;
}

void *sqread(void *) {
    std::string fname = config->output.dir + "/samples.tmp";
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

bool run_wavegen_start(int focusParam) {
    if ( thread )
        return false;
    stop = 0;
    thread = rtdo_thread_create(wglaunch, new int(focusParam), 1000000);
    return true;
}

void run_wavegen_stop() {
    stop = 1;
    rtdo_thread_join(thread);
    thread = 0;
    dlclose(lib);
    lib = 0;
}

void *wglaunch(void * arg) {
    int focusParam = *((int*)arg);
    delete (int*)arg;
    string fname = string(SOURCEDIR) + "/wavegen/WaveGen.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }
    if ( !(*(void**)(&wgmain) = dlsym(lib, "wavegen")) ) {
        cerr << dlerror() << endl;
        return (void *)EXIT_FAILURE;
    }

    if ( !config->model.load(false) ) {
        cerr << "Error: Unable to load model file '" << config->model.deffile << "'." << endl;
        return (void *)EXIT_FAILURE;
    }

    if ( focusParam == -1 ) {
        stringstream buffer;
        const vector<XMLModel::param> &p = config->model.obj->adjustableParams();
        int i = 0, j = 0;
        for ( vector<XMLModel::param>::const_iterator it = p.begin(); it != p.end() && !stop; ++it, ++i ) {
            inputSpec is = wgmain(i, config->wg.ngen);
            cout << it->name << ", best fit:" << endl;
            cout << is << endl;
            j = 0;
            for ( vector<XMLModel::param>::const_iterator jt = p.begin(); jt != p.end() && !stop; ++jt, ++j )
                buffer << (j==i ? 1 : 0) << " ";
            buffer << is << endl;
        }
        if ( !stop ) {
            string filename = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
                    + config->model.obj->name() + "_wave.";
            for ( i = 0; !access(string(filename + to_string(i)).c_str(), F_OK); i++ );
            ofstream file(filename + to_string(i));
            file << buffer.str();
            file.close();
        }
    } else {
        inputSpec is = wgmain(focusParam, config->wg.ngen);
        cout << config->model.obj->adjustableParams().at(focusParam).name << ", best fit:" << endl;
        cout << is << endl;
    }
    return 0;
}
