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
    bool valid;
    bool tested;
};
class BacklogVirtual
{
public:
    enum SortBy { ErrScore, RankScore };

    BacklogVirtual(int size, int nstims) :
        size(size), nstims(nstims), log(std::list<LogEntry>(size)) {}
    virtual ~BacklogVirtual() {}

    virtual void touch(errTupel *t, int generation, int stim, int rank = 0) = 0;
    virtual void score() = 0;
    virtual void sort(SortBy s, bool prioritiseTested = false) = 0;

    int size;
    int nstims;
    std::list<LogEntry> log;
};
class Backlog : public BacklogVirtual
{
public:
    Backlog(int size, int nstims);
    ~Backlog();
    void touch(errTupel *t, int generation, int stim, int rank = 0);
    void score();
    void sort(SortBy s, bool prioritiseTested = false);
};
}

#endif // SHARED_H
