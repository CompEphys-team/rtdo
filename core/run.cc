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
#include "util.h"
#include "run.h"
#include "config.h"
#include "xmlmodel.h"
#include "realtimeenvironment.h"

using namespace std;

bool compile_model(XMLModel::outputType type) {
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
        return false;
    }

    string modelname = config->model.obj->generateDefinition(type, popsize, INSTANCEDIR);

    string cmd = string("cd ") + INSTANCEDIR + " && " + cxxflags + "buildmodel.sh " + modelname + " 0 2>&1";
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
    string includeflags = string("INCLUDE_FLAGS='")
            + "-include " + INSTANCEDIR + "/" + modelname + "_CODE/runner.cc "
            + "-include " + INSTANCEDIR + "/" + modelname + ".cc' ";
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
        return false;
    }

    cout << "Compilation successful." << endl;
    return true;
}

bool run_vclamp(bool *stopFlag)
{
    bool stopdummy = false;
    if ( !stopFlag )
        stopFlag =& stopdummy;

    if ( !config->model.load(false) ) {
        cerr << "Error: Unable to load model file '" << config->model.deffile << "'." << endl;
        return false;
    }

    RealtimeEnvironment &env = RealtimeEnvironment::env();

    void *lib;
    int (*libmain)(const char*, const char*, const char*, bool *);
    string fname = string(SOURCEDIR) + "/simulation/VClampGA.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        std::cerr << dlerror() << endl;
        return false;
    }
    if ( !(*(void**)(&libmain) = dlsym(lib, "vclamp")) ) {
        std::cerr << dlerror() << endl;
        return false;
    }
    if ( config->model.obj->genn_float() ) {
        float (*simF)(float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env.setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double);
        if ( !(*(void**)(&simD) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env.setSimulator(simD);
    }

    env.clearChannels();
    for ( Channel &c : config->io.channels ) {
        if ( c.ID() == config->vc.in ) {
            env.addChannel(c);
            c.readOffset();
        }
        if ( c.ID() == config->vc.out ) {
            env.addChannel(c);
            c.readOffset();
        }
    }
    env.setSupersamplingRate(config->io.ai_supersampling);
    env.setDT(config->io.dt);
    env.useSimulator(false);

    libmain("live", config->output.dir.c_str(), config->vc.wavefile.c_str(), stopFlag);
    dlclose(lib);

//    // Spit out some of the best models
//    if ( logp ) {
//        void (*backlogSort)(backlog::Backlog *, bool discardUntested);
//        dlerror();
//        if ( !(*(void**)(&backlogSort) = dlsym(lib, "BacklogSort")) ) {
//            cerr << dlerror() << endl;
//            return (void *)EXIT_FAILURE;
//        }
//        cout << endl;
//        backlogSort(logp, false);
//        cout << "Backlog contains " << logp->log.size() << " valid entries, of which ";
//        backlogSort(logp, true);
//        cout << logp->log.size() << " were tested on all stimuli." << endl;
//        int i = 0;
//        for ( list<backlog::LogEntry>::iterator e = logp->log.begin(); e != logp->log.end() && i < 20; ++e, ++i ) {
//            cout << i << ": uid " << e->uid << ", since " << e->since << ", err=" << e->errScore << endl;
//            // The following code blindly assumes that (a) there is a single stimulation per parameter,
//            // and (b) the stimulations are in the same order as the parameters. This holds true for wavegen-produced
//            // stims as of 22 Jan 2016...
//            vector<double>::const_iterator errs = e->err.begin();
//            vector<double>::const_iterator vals = e->param.begin();
//            vector<int>::const_iterator ranks = e->rank.begin();
//            vector<XMLModel::param>::const_iterator names = config->model.obj->adjustableParams().begin();
//            cout << "\tParam\tValue\tError*\tRank*\t(* at the most recent stimulation)" << endl;
//            for ( ; errs != e->err.end() && vals != e->param.end() && names != config->model.obj->adjustableParams().end();
//                  ++errs, ++vals, ++names ) {
//                cout << '\t' << names->name << '\t' << *vals << '\t' << *errs << '\t' << *ranks << endl;
//            }
//        }
//    }

    return true;
}

bool run_wavegen(int focusParam, bool *stopFlag)
{
    bool stopdummy = false;
    if ( !stopFlag )
        stopFlag =& stopdummy;

    RealtimeEnvironment &env = RealtimeEnvironment::env();

    void *lib;
    inputSpec (*wgmain)(int, int, bool*);
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
        float (*simF)(float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env.setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double);
        if ( !(*(void**)(&simD) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env.setSimulator(simD);
    }

    env.setDT(config->io.dt);
    env.useSimulator(true);

    if ( focusParam == -1 ) {
        string filename = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
                + config->model.obj->name() + "_wave.";
        int i, j;
        for ( i = 0; !access(string(filename + to_string(i)).c_str(), F_OK); i++ ) {}
        filename += to_string(i);
        ofstream file(filename);

        const vector<XMLModel::param> &params = config->model.obj->adjustableParams();
        i = 0;
        for ( auto &p : params ) {
            inputSpec is = wgmain(i, config->wg.ngen, stopFlag);
            cout << p.name << ", best fit:" << endl;
            cout << is << endl;
            j = 0;
            for ( auto &_ : params ) {
                file << (j==i) << " ";
                ++j;
            }
            file << is << endl;
            file.flush();
            ++i;
            if ( *stopFlag )
                break;
        }
        file.close();
    } else {
        inputSpec is = wgmain(focusParam, config->wg.ngen, stopFlag);
        cout << config->model.obj->adjustableParams().at(focusParam).name << ", best fit:" << endl;
        cout << is << endl;
    }
    return 0;
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
void run_digest(int generation, double best_err, double mavg, int nextS) {
    cout << generation << " " << best_err << " " << mavg << " " << nextS << endl;
}
