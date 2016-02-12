/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include <algorithm>
#include "shared.h"
#include "realtimethread.h"
#include "realtimeconditionvariable.h"

using namespace std;

namespace backlog {

long long target;
bool uidMatch(LogEntry &e)
{
    return e.uid == target;
}

bool compareErrScore(const LogEntry &lhs, const LogEntry &rhs)
{
    return lhs.errScore < rhs.errScore;
}

bool compareRankScore(const LogEntry &lhs, const LogEntry &rhs)
{
    return lhs.rankScore < rhs.rankScore;
}

LogEntry::LogEntry() : valid(false) {}

LogEntry::LogEntry(int idx, int generation, int nstims) :
    param(vector<double>(NPARAM)),
    err(vector<double>(nstims, -1)),
    rank(vector<int>(nstims, 0)),
    errScore(0),
    rankScore(0),
    since(generation),
    uid(uids[idx]),
    valid(true),
    tested(true)
{
    int i = 0;
    for ( double &p : param ) {
        p = mparam[i++][idx];
    }
}

Backlog::Backlog(int size, int nstims) :
    BacklogVirtual(size, nstims)
{}

Backlog::~Backlog() {}
    
void Backlog::touch(errTupel *t, int generation, int stim, int rank)
{
    target = uids[t->id];
    list<LogEntry>::iterator it = find_if(log.begin(), log.end(), uidMatch);
    if ( it == log.end() ) {
        log.pop_front();
        log.push_back(LogEntry(t->id, generation, nstims));
        it = --log.end();
    } else {
        log.splice(log.end(), log, it);
    }
    it->err[stim] = t->err;
    it->rank[stim] = rank;
}

void Backlog::score()
{
    list<LogEntry>::iterator it = log.begin();
    while ( it != log.end() ) {
        if ( !it->valid ) {
            it = log.erase(it);
            continue;
        }
        double sumErr = 0;
        int sumRank = 0;
        int divider = nstims;
        for ( int i = 0; i < nstims; i++ ) {
            if ( it->rank.at(i) ) {
                sumErr += it->err.at(i);
                sumRank += it->rank.at(i);
            } else {
                --divider;
                it->tested = false;
            }
        }
        it->errScore = sumErr/divider;
        it->rankScore = sumRank * 1.0/divider;
        ++it;
    }
}

void Backlog::sort(SortBy s, bool prioritiseTested)
{
    switch ( s ) {
    case ErrScore:
        log.sort(compareErrScore);
        break;
    case RankScore:
        log.sort(compareRankScore);
        break;
    }
}


class AsyncLog
{
public:
    AsyncLog(BacklogVirtual *log) :
        _log(log),
        stop(false),
        waiting(false),
        t(&AsyncLog::execStatic, this, 80, 256*1024)
    {}
    
    ~AsyncLog()
    {
        halt();
    }

    void touch(errTupel *first, errTupel *last, int generation, int stim)
    {
        if ( stop ) {
            return;
        } else if ( waiting ) {
            wait();
        }
        this->first = first;
        this->last = last;
        this->generation = generation;
        this->stim = stim;
        waiting = true;
        execGo.signal();

    }

    void wait()
    {
        if ( waiting ) {
            execDone.wait();
            waiting = false;
        }
    }
    
    void halt()
    {
        if ( !stop ) {
            stop = true;
            wait();
            t.join();
        }
    }

    BacklogVirtual *log()
    {
        return _log;
    }

private:
    static void *execStatic(void *_this)
    {
        ((AsyncLog *)_this)->exec();
        return 0;
    }

    void exec()
    {
        while ( !stop ) {
            execGo.wait();
            if ( !stop ) {
                int rank = 1;
                for ( errTupel *it = first; it <= last; ++it, ++rank ) {
                    _log->touch(it, generation, stim, rank);
                }
            }
            execDone.signal();
        }
    }

    BacklogVirtual *_log;

    bool stop, waiting;

    RealtimeConditionVariable execGo, execDone;
    RealtimeThread t;

    errTupel *first, *last;
    int generation, stim;
};


} // namespace backlog


extern "C" backlog::BacklogVirtual *BacklogCreate(int size, int nstims) {
    return new backlog::Backlog(size, nstims);
}

extern "C" void BacklogDestroy(backlog::BacklogVirtual **log) {
    delete *log;
    *log = NULL;
}
