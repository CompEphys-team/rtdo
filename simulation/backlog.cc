/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include <algorithm>
#include <limits>
#include <thread>
#include <mutex>
#include <unistd.h>
#include "shared.h"

using namespace std;

namespace backlog {
numeric_limits<scalar> scalarLim;

long long target;
bool uidMatch(LogEntry &e)
{
    return e.uid == target;
}

bool compareErrScore(LogEntry &lhs, LogEntry &rhs)
{
    return lhs.errScore < rhs.errScore;
}

LogEntry::LogEntry() {}

LogEntry::LogEntry(int idx, int generation, int nstims) :
    since(generation),
    uid(uids[idx]),
    err(vector<double>(nstims, scalarLim.lowest())),
    param(vector<double>(NPARAM)),
    rank(vector<int>(nstims, 0))
{
    int i = 0;
    for ( vector<double>::iterator it = param.begin(); it != param.end(); ++it, ++i ) {
        *it = mparam[i][idx];
    }
}

Backlog::Backlog(int size, int nstims) :
    size(size),
    nstims(nstims),
    log(list<LogEntry>(size))
{}
    
void Backlog::touch(int idx, int generation, int stim, int rank)
{
    target = uids[idx];
    list<LogEntry>::iterator it = find_if(log.begin(), log.end(), uidMatch);
    if ( it == log.end() ) {
        log.pop_front();
        log.push_back(LogEntry(idx, generation, nstims));
        it = --log.end();
    } else {
        log.splice(log.end(), log, it);
    }
    it->err[stim] = errHH[idx];
    it->rank[stim] = rank;
}

void Backlog::sort(bool discardUntested)
{
    // Calculate errScore
    list<LogEntry>::iterator it = log.begin();
    while ( it != log.end() ) {
        double sumErr = 0;
        int divider = nstims;
        for ( vector<double>::iterator errIt = it->err.begin(); errIt != it->err.end(); ++errIt ) {
            if ( *errIt == scalarLim.lowest() ) {
                if ( discardUntested ) {
                    divider = 0;
                    break;
                } else {
                    --divider;
                }
            } else {
                sumErr += *errIt;
            }
        }

        if ( divider == 0 || it->err.empty() ) {
            it = log.erase(it);
        } else {
            it->errScore = sumErr/divider;
            ++it;
        }
    }

    // Sort
    log.sort(compareErrScore);
}


class AsyncLog
{
public:
    AsyncLog(int size, int nstims) :
        _log(Backlog(size, nstims)),
        stop(false),
        waiting(false)
    {
        mBarrierForExec.lock();
        mHeldByBusyExec.lock();
        t = thread(&AsyncLog::exec, this);

        // Make sure exec is fully initialised
        while ( mHeldByIdleExec.try_lock() ) {
            mHeldByIdleExec.unlock();
            usleep(5);
        }
    }
    
    ~AsyncLog()
    {
        halt();
    }

    void touch(errTupel *first, errTupel *last, int generation, int stim, int rank = 0)
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

        // Exec is on a lower priority, non-RT thread, so keeping it happy needs some care:
        // Release exec
        mBarrierForExec.unlock();
        // Make sure exec is released
        mHeldByIdleExec.lock();
        mHeldByIdleExec.unlock();
        // Lock exec's next loop
        mBarrierForExec.lock();
        // Finally, let exec do its job
        mHeldByBusyExec.unlock();
    }

    void wait()
    {
        if ( waiting ) {
            mHeldByBusyExec.lock();
            waiting = false;
        }
    }
    
    void halt()
    {
        if ( !stop ) {
            stop = true;
            wait();
            mBarrierForExec.unlock();
            mHeldByBusyExec.unlock();
            t.join();
        }
    }

    Backlog *log()
    {
        return &_log;
    }

private:
    void exec()
    {
        while ( !stop ) {
            mHeldByIdleExec.lock();
            mBarrierForExec.lock();
            mBarrierForExec.unlock();
            if ( stop ) {
                break;
            }
            mHeldByIdleExec.unlock();
            mHeldByBusyExec.lock();
            if ( !stop ) {
                int rank = 1;
                for ( errTupel *it = first; it <= last; ++it, ++rank ) {
                    _log.touch(it->id, generation, stim, rank);
                }
            }
            mHeldByBusyExec.unlock();
        }
    }

    Backlog _log;
    thread t;
    mutex mBarrierForExec, mHeldByBusyExec, mHeldByIdleExec;

    errTupel *first, *last;
    int generation, stim;

    bool stop, waiting;
};


} // namespace backlog


extern "C" backlog::Backlog *BacklogMaker(int size, int nstims) {
    return new backlog::Backlog(size, nstims);
}

extern "C" void BacklogTouch(backlog::Backlog *log, int idx, int generation, int stim) {
    log->touch(idx, generation, stim);
}

extern "C" void BacklogSort(backlog::Backlog *log, bool disqualifyUntested) {
    log->sort(disqualifyUntested);
}
