#include "runner.h"
#include "run.h"
#include "realtimethread.h"

void *Runner::launchStatic(void *_this)
{
    return ((Runner *)_this)->launch();
}

CompileRunner::CompileRunner(XMLModel::outputType type, QObject *parent) :
    Runner(parent),
    _type(type)
{}

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


bool VClampRunner::start()
{
    if ( _running )
        return false;

    _running = true;
    _stop = false;
    t.reset(new RealtimeThread(Runner::launchStatic, this, 20, 256*1024));

    return true;
}

bool VClampRunner::stop()
{
    return _stop = true;
}

void *VClampRunner::launch()
{
    bool ret = run_vclamp(&_stop);
    _running = false;
    emit processCompleted(ret);
    return 0;
}


bool WaveGenRunner::start()
{
    if ( _running )
        return false;

    _running = true;
    _stop = false;
    t.reset(new RealtimeThread(Runner::launchStatic, this, 20, 256*1024));

    return true;
}

bool WaveGenRunner::stop()
{
    return _stop = true;
}

void *WaveGenRunner::launch()
{
    bool ret = run_wavegen(-1, &_stop);
    _running = false;
    emit processCompleted(ret);
    return 0;
}
