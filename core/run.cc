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

void write_backlog(ofstream &file, backlog::BacklogVirtual *log, bool ignoreUntested)
{
    // Assumption: ordered 1-to-1 mapping from stimulations to parameters
    file << "# fully tested\tavg err\tavg rank" << endl
         << "#\t" << "param"
         << '\t' << "value"
         << '\t' << "error"
         << '\t' << "rank"
         << endl
         << endl;
    for ( backlog::LogEntry &e : log->log ) {
        if ( ignoreUntested && !e.tested )
            continue;
        file << e.tested << '\t' << e.errScore << '\t' << e.rankScore << endl;
        for ( int i = 0; i < log->nstims; i++ ) {
            file << '\t' << config->model.obj->adjustableParams().at(i).name
                 << '\t' << e.param.at(i)
                 << '\t' << e.err.at(i)
                 << '\t' << e.rank.at(i)
                 << endl;
        }
        file << endl;
    }
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

    RealtimeEnvironment* &env = RealtimeEnvironment::env();

    void *lib;

    Experiment *(*vccreate)(conf::Config *cfg, int logSize, ostream &logOutput, size_t channel, size_t nchans);
    void (*vcdestroy)(Experiment **);

    string fname = string(SOURCEDIR) + "/simulation/VClampGA.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        std::cerr << dlerror() << endl;
        return false;
    }
    if ( !(*(void**)(&vccreate) = dlsym(lib, "VClampCreate")) ) {
        std::cerr << dlerror() << endl;
        return false;
    }
    if ( !(*(void**)(&vcdestroy) = dlsym(lib, "VClampDestroy")) ) {
        std::cerr << dlerror() << endl;
        return false;
    }
    if ( config->model.obj->genn_float() ) {
        float (*simF)(float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double);
        if ( !(*(void**)(&simD) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simD);
    }

    env->clearChannels();
    for ( Channel &c : config->io.channels ) {
        if ( c.ID() == config->vc.in ) {
            c.readOffset();
            env->addChannel(c);
        }
        if ( c.ID() == config->vc.out ) {
            c.readOffset();
            env->addChannel(c);
        }
    }
    env->setSupersamplingRate(config->io.ai_supersampling);
    env->setDT(config->io.dt);
    env->useSimulator(false);

    time_t tt = time(NULL);
    char timestr[32];
    strftime(timestr, 32, "%Y%m%d-%H%M", localtime(&tt));
    string outdir = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
            + timestr + "_" + config->model.obj->name() + "_vclamp";
    if ( mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
        if ( errno == EEXIST ) {
            outdir += '-';
            int i;
            for ( i = 1; mkdir(string(outdir + to_string(i)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); i++ ) {}
            outdir += to_string(i);
        } else {
            cerr << "Could not create output directory \"" << outdir << "\": mkdir returned " << errno << endl;
            return false;
        }
    }
    cout << "All outputs saved to " << outdir << "." << endl;

    config->save(outdir + "/config_start.xml");

    {
        ifstream src(config->model.deffile);
        ofstream dest(outdir + "/model.xml");
        dest << src.rdbuf();
    }
    {
        ifstream src(config->vc.wavefile);
        ofstream dest(outdir + "/waveforms.stim");
        dest << src.rdbuf();
    }

    ofstream runtime_logf(outdir + "/runstats.log");
    runtime_logf << "# Model: " << config->model.deffile << endl;
    runtime_logf << "# Stimulations: " << config->vc.wavefile << endl;
    runtime_logf << "# DT = " << config->io.dt << endl;
    runtime_logf << "# " << config->io.ai_supersampling << " analog samples per DT" << endl;
    runtime_logf << "# " << config->model.cycles << " simulation cycles per DT" << endl;
    runtime_logf << "# Clamp gain: " << config->vc.gain << " V/V, E2 resistance: " << config->vc.resistance << " MOhm" << endl;
    runtime_logf << "#" << endl;
    // Header for output as defined in backlog::BacklogVirtual::exec():
    runtime_logf << "# Epoch\tStimulation\tBest error\tMean error\terror SD";
    for ( const XMLModel::param &p : config->model.obj->adjustableParams() ) {
        runtime_logf << "\tBest model " << p.name << "\tMean " << p.name << "\t" << p.name << " SD";
    }
    runtime_logf << endl;
    teestream tee(runtime_logf, cout);

    {
        // Run!
        Experiment *vc = vccreate(config, config->vc.popsize, tee, 0, 1);
        vc->initModel();
        vc->run(stopFlag);

        config->save(outdir + "/config_end.xml");

        // Dump final model set
        shared_ptr<backlog::BacklogVirtual> logp = vc->log();
        logp->score();
        logp->sort(backlog::BacklogVirtual::ErrScore, true);
        ofstream models_logf(outdir + "/models.log");
        write_backlog(models_logf, &*logp, false);
        models_logf.close();

        int tested = 0;
        for ( backlog::LogEntry &e : logp->log ) {
            tested += (int)e.tested;
        }

        vcdestroy(&vc);

        runtime_logf.close();

        cout << endl << "Fitting complete. " << tested << " models fully evaluated." << endl;
        cout << "All outputs saved to " << outdir << "." << endl;

    }

    dlclose(lib);
    return true;
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
        float (*simF)(float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double);
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
        float (*simF)(float*, float*, float);
        if ( !(*(void**)(&simF) = dlsym(lib, "simulateSingleNeuron")) ) {
            std::cerr << dlerror() << endl;
            return false;
        }
        env->setSimulator(simF);

    } else {
        double (*simD)(double*, double*, double);
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
