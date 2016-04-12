/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-03-14

--------------------------------------------------------------------------*/
#ifndef MODULE_H
#define MODULE_H

#include <QObject>
#include <functional>
#include <deque>
#include <memory>
#include "experiment.h"
#include "realtimethread.h"
#include "realtimeconditionvariable.h"

class Module : public QObject
{
    Q_OBJECT
public:
    Module(QObject *parent = nullptr);
    ~Module();

    //!< Queue @arg fn to run on the Module thread. Returns a unique handle to the queued function, which is also passed to it on execution.
    //! @arg fn will typically be a lambda function containing a call to an Experiment function.
    //! @arg logEntry is a human-readable string used in log entries.
    int push(std::string logEntry, std::function<void(int handle)> fn);

    //!< Erase the action referred to by @arg handle. Can only affect actions that have not yet started.
    //! @return true if an action has been removed
    bool erase(int handle);

    bool busy();
    size_t qSize();

    Experiment *vclamp;

    std::string outdir;

public slots:
    //!< Start executing queued actions
    void start();

    //!< Interrupt the current action, discarding any queued functions. Returns immediately without waiting for the action to complete.
    //! New actions can be queued immediately and will execute as soon as the current action returns.
    void stop();

    //!< Interrupt the current action, continuing execution of the queue. Returns immediately without waiting for results.
    //! Effectively, this only raises the Experiment stopFlag, so actions that do not depend on this will continue uninterrupted.
    void skip();

signals:
    void complete(int handle);
    void outdirSet();

private:
    static void *execStatic(void *);
    void exec();
    bool initOutput();

    struct action
    {
        int handle;
        std::string logEntry;
        std::function<void(int)> fn;

        action(int h, std::string l, std::function<void(int)> f) :
            handle(h),
            logEntry(l),
            fn(f)
        {}
    };

    void *lib;

    std::deque<action> q;

    int handle_ctr;
    bool firstrun;

    RealtimeConditionVariable sem, lock;
    bool _exit, _stop, _busy;
    std::unique_ptr<RealtimeThread> t;
};

#endif // MODULE_H
