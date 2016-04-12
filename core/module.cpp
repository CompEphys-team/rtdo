/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-03-14

--------------------------------------------------------------------------*/
#include "module.h"
#include <dlfcn.h>
#include <ctime>
#include <sys/stat.h>
#include <stdexcept>
#include <fstream>
#include "run.h"
#include "realtimeenvironment.h"
#include "teestream.h"

Module::Module(QObject *parent) :
    QObject(parent),
    vclamp(nullptr),
    lib(nullptr),
    handle_ctr(0),
    firstrun(true),
    _exit(false),
    _stop(true),
    _busy(false)
{
    if ( !config->model.load(false) ) {
        string err = string("Unable to load model file '") + config->model.deffile + "'.";
        throw runtime_error(err);
    }

    string fname = string(SOURCEDIR) + "/simulation/VClampGA.so";
    dlerror();
    if ( ! (lib = dlopen(fname.c_str(), RTLD_NOW)) ) {
        string err(dlerror());
        throw runtime_error(err);
    }

    Experiment *(*vccreate)(conf::Config *cfg, int logSize, ostream &logOutput, size_t channel, size_t nchans);
    if ( !(*(void**)(&vccreate) = dlsym(lib, "VClampCreate")) ) {
        string err(dlerror());
        dlclose(lib);
        throw runtime_error(err);
    }

    vclamp = vccreate(config, config->vc.popsize, cout, 0, 1);
    vclamp->initModel();

    // Start thread at the end, when all possible pitfalls are avoided
    t.reset(new RealtimeThread(Module::execStatic, this, config->rt.prio_module, config->rt.ssz_module, config->rt.cpus_module));
    lock.signal();
}

bool Module::initOutput()
{
    time_t tt = time(NULL);
    char timestr[32];
    strftime(timestr, 32, "%Y%m%d-%H%M", localtime(&tt));
    outdir = config->output.dir + (config->output.dir.back()=='/' ? "" : "/")
            + timestr + "_" + config->model.obj->name();
    if ( mkdir(outdir.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) ) {
        if ( errno == EEXIST ) {
            outdir += '-';
            int i;
            for ( i = 1; mkdir(string(outdir + to_string(i)).c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH); i++ ) {}
            outdir += to_string(i);
        } else {
            cerr << "Could not create output directory \"" << outdir << "\": mkdir returned " << errno << endl;
            cerr << "Make sure the output path exists with appropriate permissions and try again." << endl;
            return false;
        }
    }
    cout << "All outputs saved to " << outdir << "." << endl;

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

    return true;
}

Module::~Module()
{
    _exit = true;
    stop();
    sem.broadcast();
    t->join();
    vclamp->log()->wait();

    void (*vcdestroy)(Experiment **);
    if ( !(*(void**)(&vcdestroy) = dlsym(lib, "VClampDestroy")) ) {
        string err(dlerror());
        throw runtime_error(err);
    }
    vcdestroy(&vclamp);
    dlclose(lib);
}

void *Module::execStatic(void *_this)
{
    ((Module *)_this)->exec();
    return nullptr;
}

void Module::exec()
{
    ofstream actionLog(outdir + "/actions.log");

    while ( !_exit ) {
        sem.wait();

        lock.wait(); // Locked until vvvvvvvvvvvvvvvvvv

        if ( firstrun && !_stop && q.size() ) {
            if ( !initOutput() ) {
                _stop = true;
            } else {
                firstrun = false;
                emit outdirSet();
            }
        }

        _busy = true;
        while ( !_stop && q.size() ) { //    while
            vclamp->stopFlag = false;
            action p = q.front();
            q.pop_front();
            lock.signal(); // Lock ends ^^^^^^^^^^^^^^^

            // Tee output to new logfile
            ofstream logf(outdir + "/" + to_string(p.handle) + "_results.log");
            teestream tee(logf, cout);
            vclamp->log()->wait();
            vclamp->log()->out->flush();
            vclamp->log()->out =& tee;
            tee << "# Action " << p.handle << ": <" << p.logEntry << "> begins..." << endl;

            // Save config
            config->save(outdir + "/" + to_string(p.handle) + "_config.xml");

            // Set up I/O
            RealtimeEnvironment *&env = RealtimeEnvironment::env();
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

            // Invoke
            p.fn(p.handle);

            string status = vclamp->stopFlag ? "ended on user request" : "completed normally";
            actionLog << p.handle << '\t' << p.logEntry << '\t' << status << endl;

            // Wait for backlog completion and revoke tee
            vclamp->log()->wait();
            tee << "# Action " << p.handle << ": <" << p.logEntry << "> " << status << endl;
            tee.flush();
            vclamp->log()->out =& cout;
            // tee and logf destroyed on loop end

            lock.wait(); // Locked until vvvvvvvvvvvvvvvvvvvvvvv
            emit complete(p.handle);
        } //                            end while
        _stop = true;
        _busy = false;
        lock.signal(); // Lock ends ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    }
}

int Module::push(std::string logEntry, std::function<void(int)> fn)
{
    lock.wait();
    int handle = ++handle_ctr;
    q.push_back(action(handle, logEntry, fn));
    lock.signal();

    return handle;
}

bool Module::erase(int handle)
{
    lock.wait();
    bool ret = false;
    for ( auto p = q.begin(); p != q.end(); ++p ) {
        if ( p->handle == handle ) {
            q.erase(p);
            ret = true;
            break;
        }
    }
    lock.signal();

    return ret;
}

bool Module::busy()
{
    lock.wait();
    bool ret = _busy;
    lock.signal();

    return ret;
}

size_t Module::qSize()
{
    lock.wait();
    size_t ret = q.size();
    lock.signal();

    return ret;
}

void Module::start()
{
    lock.wait();
    if ( !_busy && q.size() ) {
        _stop = false;
        sem.signal();
    }
    lock.signal();
}

void Module::stop()
{
    lock.wait();
    vclamp->stopFlag = true;
    _stop = true;
    q.clear();
    lock.signal();
}

void Module::skip()
{
    lock.wait();
    vclamp->stopFlag = true;
    lock.signal();
}


