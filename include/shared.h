/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-21

--------------------------------------------------------------------------*/
#ifndef SHARED_H
#define SHARED_H

#include <iostream>
#include <queue>
#include <vector>
#include <list>
#include "realtimeconditionvariable.h"
#include "realtimethread.h"

struct inputSpec {
    double t;
    double ot;
    double dur;
    double baseV;
    int N;
    std::vector<double> st;
    std::vector<double> V;
    double fit;
};
std::ostream &operator<<(std::ostream &os, inputSpec &I);
std::istream &operator>>(std::istream &is, inputSpec &I);

struct errTupel
{
    unsigned int id;
    double err;
};

enum stageEnum {
    stDetuneAdjust,
    stNoveltySearch,
    stWaveformOptimise,
    stObservationWindow
};

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
    long long uid;
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
        t(&BacklogVirtual::execStatic, this, 80, 256*1024)
    {}
    virtual ~BacklogVirtual() {}

    virtual void touch(errTupel *first, errTupel *last, int generation, int stim) = 0;
    virtual void score() = 0;
    virtual void sort(SortBy s, bool prioritiseTested = false) = 0;
    virtual void wait() = 0;
    virtual void halt() = 0;

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
    void halt();
};
}

#endif // SHARED_H
