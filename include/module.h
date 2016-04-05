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
#include <queue>
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

    //!< Queue @arg fn to run on the Module thread. Returns a unique handle to the queued function. Queued functions execute automatically.
    //! @arg fn will typically be a lambda function containing a call to an Experiment function
    int push(std::function<void(int handle)> fn);

    Experiment *vclamp;

    std::string outdir;

public slots:
    //!< Interrupt the current action, discarding any queued functions. Returns immediately without waiting for the action to complete.
    //! New actions can be queued immediately and will execute as soon as the current action returns.
    void stop();

signals:
    void complete(int handle);

private:
    static void *execStatic(void *);
    void exec();

    void *lib;

    std::queue<std::pair<int, std::function<void(int)>>> q;

    int handle_ctr;

    RealtimeConditionVariable sem;
    bool _exit, _stop;
    std::unique_ptr<RealtimeThread> t;
};

#endif // MODULE_H
