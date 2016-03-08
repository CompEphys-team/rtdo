/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-26

--------------------------------------------------------------------------*/
#ifndef BACKLOG_H
#define BACKLOG_H

#include "shared.h"
#include "config.h"

namespace backlog {

class LogEntry
{
public:
    LogEntry();
    LogEntry(int idx, int generation, int nstims);
    std::vector<double> param;
    std::vector<double> err;
    std::vector<int> rank;
    double errScore;
    double rankScore;
    int since;
    unsigned long long uid;
    bool tested;
};

class BacklogVirtual
{
public:
    enum SortBy { ErrScore, RankScore };

    BacklogVirtual(int size, int nstims, std::ostream &runtime_log) :
        size(size),
        nstims(nstims),
        log(std::list<LogEntry>(0)),
        out(runtime_log),
        stop(false),
        waiting(false),
        t(&BacklogVirtual::execStatic, this, config->rt.prio_backlog, config->rt.ssz_backlog, config->rt.cpus_backlog)
    {}
    virtual ~BacklogVirtual() {}

    virtual void touch(errTupel *first, errTupel *last, int generation, int stim) = 0;
    virtual void score() = 0;
    virtual void sort(SortBy s, bool prioritiseTested = false) = 0;
    virtual void wait() = 0;

    int size;
    int nstims;
    std::list<LogEntry> log;
    std::ostream &out;

protected:
    static void *execStatic(void *_this);
    void exec();

    bool stop, waiting;

    RealtimeConditionVariable execGo, execDone;
    RealtimeThread t;

    errTupel *first, *last;
    int generation, stim;
};

class Backlog : public BacklogVirtual
{
public:
    Backlog(int size, int nstims, std::ostream &runtime_log);
    ~Backlog();
    void touch(errTupel *first, errTupel *last, int generation, int stim);
    void score();
    void sort(SortBy s, bool prioritiseTested = false);
    void wait();
};

}

#endif // BACKLOG_H

