/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-01

--------------------------------------------------------------------------*/
#ifndef RUNNER_H
#define RUNNER_H

#include <QObject>
#include "realtimethread.h"
#include "realtimeconditionvariable.h"
#include "xmlmodel.h"

/** @brief Runner is a helper to simplify the management of work threads. A thread of the appropriate type is launched
 * through @fn Runner::start and can be interrupted using @fn Runner::stop. Upon completion, a
 * @fn Runner::processCompleted(bool) signal is emitted. Alternatively, you can @fn Runner::wait for completion,
 * which puts the caller on hold until the work is done.
 **/
class Runner : public QObject
{
    Q_OBJECT
public:
    explicit Runner(XMLModel::outputType type, QObject *parent = 0);
    inline bool running() const { return _running; }

signals:
    void processCompleted(bool successfully);

public slots:
    virtual bool start();
    virtual bool stop();
    virtual void wait();

protected:
    virtual void *launch();
    static void *launchStatic(void *_this);
    friend class RealtimeThread::Impl;

    std::unique_ptr<RealtimeThread> t;

    RealtimeConditionVariable _sem;

    XMLModel::outputType _type;

    bool _running;
    bool _stop;
};


//!< @brief CompileRunner is a compilation helper. Runs in the calling thread because of RTAI shenanigans.
class CompileRunner : public Runner
{
    Q_OBJECT
public:
    inline explicit CompileRunner(XMLModel::outputType type = XMLModel::WaveGen, QObject *parent = 0) : Runner(type, parent) {}

    inline void setType(XMLModel::outputType type) { _type = type; }
    inline XMLModel::outputType type() const { return _type; }

public slots:
    bool start();
    inline bool stop() { return false; }
    inline void wait() {}

protected:
    inline void *launch() { return 0; }
};

#endif // RUNNER_H
