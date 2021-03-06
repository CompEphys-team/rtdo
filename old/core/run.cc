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
#include <unistd.h>
#include <ctime>
#include "util.h"
#include "run.h"
#include "config.h"
#include "xmlmodel.h"
#include "realtimeenvironment.h"
#include "teestream.h"
#include "wavegenNS.h"
#include "backlog.h"
#include <sys/stat.h>
#include "experiment.h"

using namespace std;

bool compile_model(XMLModel::outputType type) {
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
    case XMLModel::WaveGenNoveltySearch:
        popsize = config->wg.popsize;
        simulator += "/wavegenNS";
        break;
    }

    if ( !config->model.load() ) {
        cerr << "Error: Unable to load model file '" << config->model.deffile << "'." << endl;
        return false;
    }

    string modelname = config->model.obj->generateDefinition(type, popsize, INSTANCEDIR);
    string modelpath = string(INSTANCEDIR) + "/" + modelname;

    string cmd = string("cd ") + INSTANCEDIR + " && " + "genn-buildmodel.sh -s " + modelname + ".cc 2>&1";
    cout << cmd << endl;
    FILE *is = popen(cmd.c_str(), "r");
    char buffer[1024];
    while ( fgets(buffer, 1024, is) ) {
        cout << buffer;
        if ( strstr(buffer, "error") || strstr(buffer, "Error") )
            ret = 1;
    }
    pclose(is);
    if ( ret ) {
        cerr << "Model build failed." << endl;
        return false;
    }

    // Add simulator code to the model definition file
    config->model.obj->generateSimulator(type, INSTANCEDIR);

    // Compile it all
    cmd = string("cd ") + simulator
            + " && make clean SIM_CODE=" + modelpath + "_CODE"
            + " && INCLUDE_FLAGS='"
                + "-include " + modelpath + "_CODE/definitions.h "
                + "-include " + modelpath + ".cc'"
            + " make SIM_CODE=" + modelpath + "_CODE";
#ifdef _DEBUG
    cmd += " DEBUG=1";
#endif
    cout << cmd << endl;
    if ( (ret = system(cmd.c_str())) ) {
        cerr << "Compilation failed." << endl;
        return false;
    }

    cout << "Compilation successful." << endl;
    return true;
}

void write_backlog(ofstream &file, const std::vector<const backlog::LogEntry *> sorted, bool ignoreUntested)
{
    // Assumption: ordered 1-to-1 mapping from stimulations to parameters
    file << "# fully tested\tfinal rank\tavg err\tavg rank";
    for ( auto p : config->model.obj->adjustableParams() ) {
        file << '\t' << p.name << " value"
             << '\t' << p.name << " err"
             << '\t' << p.name << " rank";
    }
    file << endl;
    int finalRank = 0;
    for ( auto *e : sorted ) {
        if ( ignoreUntested && !e->tested )
            continue;
        file << e->tested << '\t' << ++finalRank << '\t' << e->errScore << '\t' << e->rankScore;
        for ( size_t i = 0; i < config->model.obj->adjustableParams().size(); i++ ) {
            file << '\t' << e->param.at(i)
                 << '\t' << e->err.at(i)
                 << '\t' << e->rank.at(i);
        }
        file << endl;
    }
}

std::vector<double> read_model_dump(ifstream &file, int rank)
{
    // Extract a parameter set from output of @fn write_backlog. Comes with no guarantees...
    char buffer[1024];
    file.getline(buffer, 1024);
    double tmp = 0;
    file >> tmp >> tmp;
    while ( file.good() && tmp != rank ) {
        file.getline(buffer, 1024);
        file >> tmp >> tmp;
    }

    vector<double> ret(config->model.obj->adjustableParams().size());
    if ( !file.good() )
        return ret;
    file >> tmp >> tmp;
    for ( size_t i = 0; i < ret.size(); i++ ) {
        file >> ret[i] >> tmp >> tmp;
    }
    return ret;
}

bool run_wavegen(int focusParam, bool *stopFlag)
{
    bool stopdummy = false;
    if ( !stopFlag )
        stopFlag =& stopdummy;

    RealtimeEnvironment* &env = RealtimeEnvironment::env();

    void *lib;
    inputSpec (*wgmain)(conf::Config*, int, bool*);
    string fname = string(SOURCEDIR) + "/wavegen/WaveGen.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        cerr << dlerror() << endl;
        return false;
    }
    if ( !(*(void**)(&wgmain) = dlsym(lib, "wavegen")) ) {
        cerr << dlerror() << endl;
        return false;
    }

    if ( !config->model.load(false) ) {
        cerr << "Error: Unable to load model file '" << config->model.deffile << "'." << endl;
        return false;
    }
    if ( config->model.obj->genn_float() ) {
        float (*simF)(float*, float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double*, double);
        if ( !(*(void**)(&simD) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simD);
    }

    env->setDT(config->io.dt);
    env->useSimulator(true);

    if ( focusParam == -1 ) {
        string filename = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
                + config->model.obj->name() + "_wave.";
        int i;
        for ( i = 0; !access(string(filename + to_string(i)).c_str(), F_OK); i++ ) {}
        filename += to_string(i);
        ofstream file(filename);

        const vector<XMLModel::param> params = config->model.obj->adjustableParams(); // Copy to guard against user interference
        i = 0;
        for ( auto &p : params ) {
            inputSpec is = wgmain(config, i, stopFlag);
            cout << p.name << ", best fit:" << endl;
            cout << is << endl;
            for ( int j = 0; j < (int) params.size(); j++ ) {
                file << (j==i) << " ";
            }
            for ( int j = 0; j < (int) params.size(); j++ ) {
                file << "1.0 "; // sigma adjustment hotfix
            }
            file << is << endl;
            file.flush();
            ++i;
            if ( *stopFlag )
                break;
        }
        file.close();
    } else {
        inputSpec is = wgmain(config, focusParam, stopFlag);
        cout << config->model.obj->adjustableParams().at(focusParam).name << ", best fit:" << endl;
        cout << is << endl;
    }

    dlclose(lib);

    return true;
}

bool run_wavegen_NS(bool *stopFlag)
{
    bool stopdummy = false;
    if ( !stopFlag )
        stopFlag =& stopdummy;

    RealtimeEnvironment* &env = RealtimeEnvironment::env();

    if ( !config->model.load(false) ) {
        cerr << "Error: Unable to load model file '" << config->model.deffile << "'." << endl;
        return false;
    }

    void *lib;
    WavegenNSVirtual *(*wgcreate)(conf::Config *);
    void (*wgdestroy)(WavegenNSVirtual **);
    string fname = string(SOURCEDIR) + "/wavegenNS/WaveGen.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        cerr << dlerror() << endl;
        return false;
    }
    if ( !(*(void**)(&wgcreate) = dlsym(lib, "WavegenCreate")) ) {
        cerr << dlerror() << endl;
        return false;
    }
    if ( !(*(void**)(&wgdestroy) = dlsym(lib, "WavegenDestroy")) ) {
        cerr << dlerror() << endl;
        return false;
    }
    if ( config->model.obj->genn_float() ) {
        float (*simF)(float*, float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double*, double);
        if ( !(*(void**)(&simD) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simD);
    }

    env->setDT(config->io.dt);
    env->useSimulator(true);

    time_t tt = time(NULL);
    char timestr[32];
    strftime(timestr, 32, "%Y%m%d-%H%M", localtime(&tt));

    stringstream header;
    header << "# Model: " << config->model.deffile << endl
           << "# DT = " << config->io.dt << endl
           << "# " << config->model.cycles << " simulation cycles per DT" << endl
           << "# Clamp gain: " << config->vc.gain << " V/V, E2 resistance: " << config->vc.resistance << " MOhm" << endl
           << "#MATLAB headerLines = 6" << endl
           << endl;

    string wavefile_str = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
            + config->model.obj->name() + "_" + timestr + "_wavegenNS.stim";
    ofstream wavefile(wavefile_str);
    wavefile << header.str();

    string currentfile_str = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
            + config->model.obj->name() + "_" + timestr + "_wavegenNS_currents.log";
    ofstream currentfile(currentfile_str);
    currentfile << header.str();

    WavegenNSVirtual *wg = wgcreate(config);
    wg->runAll(wavefile, currentfile, stopFlag);
    wgdestroy(&wg);
    dlclose(lib);
    cout << endl << "Waveforms written to " << wavefile_str << endl;
    cout << "Current traces written to " << currentfile_str << endl;
    cout << "Waveform generation complete." << endl;

    wavefile.close();
    currentfile.close();
    return true;
}

/*
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
*/
