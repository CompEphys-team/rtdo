/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-03-14

--------------------------------------------------------------------------*/
#include "module.h"
#include <ctime>
#include <sys/stat.h>
#include <stdexcept>
#include <fstream>
#include "run.h"
#include "realtimeenvironment.h"
#include "teestream.h"

template <class T>
Module<T>::Module(QObject *parent) :
    VirtualModule(parent),
    obj(nullptr),
    outdir(""),
    lib(nullptr),
    handle_ctr(0),
    firstrun(true),
    _append(false),
    _exit(false),
    _stop(true),
    _busy(false)
{
    if ( !config->model.load(false) ) {
        string err = string("Unable to load model file '") + config->model.deffile + "'.";
        throw runtime_error(err);
    }

    lib = T::openLibrary();
    obj = T::create(lib);

    // Start thread at the end, when all possible pitfalls are avoided
    t.reset(new RealtimeThread(Module::execStatic, this, config->rt.prio_module, config->rt.ssz_module, config->rt.cpus_module));
    lock.signal();
}

template <class T>
bool Module<T>::initOutput()
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

    copyFiles();

    return true;
}

template <class T>
void Module<T>::copyFiles()
{
    ifstream src(config->model.deffile);
    ofstream dest(outdir + "/model.xml");
    dest << src.rdbuf();
}

template<>
void Module<Experiment>::copyFiles()
{
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
}

template <class T>
Module<T>::~Module()
{
    _exit = true;
    stop();
    sem.broadcast();
    t->join();
    obj->setLog(&cout);

    T::destroy(lib, &obj);
    T::closeLibrary(lib);
}

template <class T>
void *Module<T>::execStatic(void *_this)
{
    ((Module *)_this)->exec();
    return nullptr;
}

template <class T>
void Module<T>::exec()
{
    ofstream actionLog;

    while ( !_exit ) {
        sem.wait();

        lock.wait(); // Locked until vvvvvvvvvvvvvvvvvv

        if ( _append && firstrun ) {
            firstrun = false;
            actionLog.open(outdir + "/actions.log", ios_base::out | ios_base::app);
            emit outdirSet();
        }
        if ( firstrun && !_stop && q.size() ) {
            if ( !initOutput() ) {
                _stop = true;
            } else {
                firstrun = false;
                actionLog.open(outdir + "/actions.log");
                emit outdirSet();
            }
        }

        _busy = true;
        while ( !_stop && q.size() ) { //    while
            obj->stopFlag = false;
            action p = q.front();
            q.pop_front();
            lock.signal(); // Lock ends ^^^^^^^^^^^^^^^

            // Tee output to new logfile
            ofstream logf(outdir + "/" + to_string(p.handle) + "_results.log");
            teestream tee(logf, cout);
            obj->setLog(&tee);
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

            string status = obj->stopFlag ? "ended on user request" : "completed normally";
            actionLog << p.handle << '\t' << p.logEntry << '\t' << status << endl;

            // Reset output stream
            stringstream closing;
            closing << "# Action " << p.handle << ": <" << p.logEntry << "> " << status << endl;
            obj->setLog(&cout, closing.str());
            // tee and logf destroyed on loop end

            lock.wait(); // Locked until vvvvvvvvvvvvvvvvvvvvvvv
            emit complete(p.handle);
        } //                            end while
        _stop = true;
        _busy = false;
        lock.signal(); // Lock ends ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    }
}

template <class T>
int Module<T>::push(std::string logEntry, std::function<void(int)> fn)
{
    lock.wait();
    int handle = ++handle_ctr;
    q.push_back(action(handle, logEntry, fn));
    lock.signal();

    return handle;
}

template <class T>
bool Module<T>::erase(int handle)
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

template <class T>
bool Module<T>::append(string directory)
{
    if ( !firstrun || handle_ctr > 0 )
        return false;
    if ( directory.back() == '/' )
        directory.erase(directory.end()-1);

    char buffer[1024];
    unsigned int last = 0;
    ifstream oldLog(directory + "/actions.log");
    while ( oldLog.good() ) {
        oldLog.getline(buffer, 1024);
        sscanf(buffer, "%u ", &last);
    }
    if ( !last )
        return false;

    handle_ctr = last;
    _append = true;
    outdir = directory;
    push(string("Append output to ") + directory, [](int){});
    return true;
}

template <class T>
bool Module<T>::busy()
{
    lock.wait();
    bool ret = _busy;
    lock.signal();

    return ret;
}

template <class T>
size_t Module<T>::qSize()
{
    lock.wait();
    size_t ret = q.size();
    lock.signal();

    return ret;
}

template <class T>
void Module<T>::start()
{
    lock.wait();
    if ( !_busy && q.size() ) {
        _stop = false;
        sem.signal();
    }
    lock.signal();
}

template <class T>
void Module<T>::stop()
{
    lock.wait();
    obj->stopFlag = true;
    _stop = true;
    q.clear();
    lock.signal();
}

template <class T>
void Module<T>::skip()
{
    lock.wait();
    obj->stopFlag = true;
    lock.signal();
}

template class Module<Experiment>;
template class Module<WavegenNSVirtual>;
