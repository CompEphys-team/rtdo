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

namespace backlog {
class LogEntry
{
public:
    LogEntry();
    LogEntry(int idx, int generation, int nstims);
    std::vector<double> param;
    std::vector<double> err;
    double errScore;
    int since;
    long long uid;
};
class Backlog
{
public:
    Backlog(int size, int nstims);
    void touch(int idx, int generation, int stim);
    void sort(bool discardUntested = true);

    int size;
    int nstims;
    std::list<LogEntry> log;
};
}

#endif // SHARED_H
