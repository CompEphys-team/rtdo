/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-01

--------------------------------------------------------------------------*/
#include "runner.h"
#include "run.h"
#include "realtimethread.h"
#include "realtimeenvironment.h"

Runner::Runner(XMLModel::outputType type, QObject *parent) :
    QObject(parent),
    _type(type),
    _running(false),
    _stop(false)
{}

void *Runner::launchStatic(void *_this)
{
    void *ret = ((Runner *)_this)->launch();
    ((Runner *)_this)->_sem.broadcast();
    return ret;
}

void Runner::wait()
{
    if ( _running )
        _sem.wait();
}

bool Runner::start()
{
    if ( _running )
        return false;

    _running = true;
    _stop = false;
    t.reset(new RealtimeThread(Runner::launchStatic, this, 20, 256*1024));

    return true;
}

bool Runner::stop()
{
    return _stop = true;
}

void *Runner::launch()
{
    bool ret = false;
    try {
        switch ( _type ) {
        case XMLModel::VClamp:
            ret = run_vclamp(&_stop);
            break;
        case XMLModel::WaveGen:
            ret = run_wavegen(-1, &_stop);
            break;
        case XMLModel::WaveGenNoveltySearch:
            ret = run_wavegen_NS(&_stop);
            break;
        }
    } catch ( RealtimeException &e ) {
        std::cerr << "An exception occurred in the worker thread: " << e.what() << std::endl;
        RealtimeEnvironment::env().pause();
    }
    _running = false;
    emit processCompleted(ret);
    return 0;
}


bool CompileRunner::start()
{
    if ( _running )
        return false;

    _running = true;
    bool ret = compile_model(_type);
    emit processCompleted(ret);
    _running = false;

    return true;
}
