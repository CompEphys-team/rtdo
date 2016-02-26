/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include <algorithm>
#include "backlog.h"
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

bool compareTested(const LogEntry &lhs, const LogEntry &rhs)
{
    return lhs.tested && !rhs.tested;
}

LogEntry::LogEntry() : // Default construction for dummy entries, i.e. mean/sd in BacklogVirtual::exec
    param(vector<double>(NPARAM, 0)),
    err(vector<double>(1, 0))
{}

LogEntry::LogEntry(int idx, int generation, int nstims) :
    param(vector<double>(NPARAM)),
    err(vector<double>(nstims, -1)),
    rank(vector<int>(nstims, 0)),
    errScore(0),
    rankScore(0),
    since(generation),
    uid(uids[idx]),
    tested(true)
{
    int i = 0;
    for ( double &p : param ) {
        p = mparam[i++][idx];
    }
}

Backlog::Backlog(int size, int nstims, ostream &runtime_log) :
    BacklogVirtual(size, nstims, runtime_log)
{}

Backlog::~Backlog() {}
    
void Backlog::touch(errTupel *first, errTupel *last, int generation, int stim)
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

void Backlog::score()
{
    list<LogEntry>::iterator it = log.begin();
    while ( it != log.end() ) {
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

    if ( prioritiseTested ) {
        log.sort(compareTested);
    }
}

void Backlog::wait()
{
    if ( waiting ) {
        execDone.wait();
        waiting = false;
    }
}

void Backlog::halt()
{
    if ( !stop ) {
        stop = true;
        wait();
        t.join();
    }
}

void *BacklogVirtual::execStatic(void *_this)
{
    ((BacklogVirtual*)_this)->exec();
    return 0;
}

void BacklogVirtual::exec()
{
    while ( !stop ) {
        execGo.wait();
        if ( !stop ) {
            list<LogEntry> tmp;
            LogEntry mean, sd;
            int rank = 1;
            for ( errTupel *et = first; et <= last; ++et, ++rank ) {
                // Enter new entry / refresh existing entry
                target = uids[et->id];
                list<LogEntry>::iterator it = find_if(log.begin(), log.end(), uidMatch);
                if ( it == log.end() ) {
                    tmp.push_back(LogEntry(et->id, generation, nstims));
                    it = --tmp.end();
                } else {
                    tmp.splice(tmp.end(), log, it);
                }
                it->err[stim] = et->err;
                it->rank[stim] = rank;

                // Tally up for mean/SD
                for ( int i = 0; i < NPARAM; i++ ) {
                    mean.param[i] += it->param[i];
                }
                mean.err[0] += et->err;
            }

            // Calculate parameter & error mean/SD
            for ( int i = 0; i < NPARAM; i++ ) {
                mean.param[i] /= (rank-1);
            }
            mean.err[0] /= (rank-1);
            for ( LogEntry &le : tmp ) {
                double diff;
                for ( int i = 0; i < NPARAM; i++ ) {
                    diff = mean.param[i] - le.param[i];
                    sd.param[i] += (diff*diff);
                }
                diff = mean.err[0] - le.err[stim];
                sd.err[0] = (diff*diff);
            }
            for ( int i = 0; i < NPARAM; i++ ) {
                sd.param[i] = sqrt(sd.param[i] / (rank-1));
            }
            sd.err[0] = sqrt(sd.err[0] / (rank-1));

            // Dump error & param mean/SD to file
            out << generation << '\t' << stim << '\t' << tmp.begin()->err[stim] << '\t' << mean.err[0] << '\t' << sd.err[0];
            for ( int i = 0; i < NPARAM; i++ ) {
                out << '\t' << tmp.begin()->param[i] << '\t' << mean.param[i] << '\t' << sd.param[i];
            }
            out << endl;

            // Replace outdated log with tmp
            log.clear();
            log.splice(log.begin(), tmp);
        }
        execDone.signal();
    }
}


} // namespace backlog


extern "C" backlog::BacklogVirtual *BacklogCreate(int size, int nstims, ostream &out) {
    return new backlog::Backlog(size, nstims, out);
}

extern "C" void BacklogDestroy(backlog::BacklogVirtual **log) {
    delete *log;
    *log = NULL;
}
