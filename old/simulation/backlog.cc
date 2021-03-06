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

bool prioritiseTested;
bool compareErrScore(const LogEntry *lhs, const LogEntry *rhs)
{
    if ( prioritiseTested && lhs->tested != rhs->tested ) {
        return lhs->tested;
    }
    if ( lhs->errScore != rhs->errScore )
        return lhs->errScore < rhs->errScore;
    else
        return lhs->rankScore < rhs->rankScore;
}

bool compareRankScore(const LogEntry *lhs, const LogEntry *rhs)
{
    if ( prioritiseTested && lhs->tested != rhs->tested ) {
        return lhs->tested; // => !rhs->tested
    }
    if ( lhs->rankScore != rhs->rankScore )
        return lhs->rankScore < rhs->rankScore;
    else // rankScores are equal, fall back to errScore
        return lhs->errScore < rhs->errScore;
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
    uid(currentExperiment->models[idx].uid),
    tested(true),
    idx(idx)
{
    int i = 0;
    for ( double &p : param ) {
        p = mparam[i++][idx];
    }
}

Backlog::Backlog(int size, int nstims, ostream *runtime_log) :
    BacklogVirtual(size, nstims, runtime_log)
{}

Backlog::~Backlog()
{
    stop = true;
    execGo.broadcast();
    t.join();
}
    
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

std::vector<const LogEntry *> Backlog::sort(SortBy s, bool _prioritiseTested) const
{
    std::vector<const LogEntry*> ret(log.size());
    auto rit = ret.begin();
    auto lit = log.begin();
    for ( ; lit != log.end(); ++rit, ++lit ) {
        *rit = &*lit;
    }

    prioritiseTested = _prioritiseTested;
    switch ( s ) {
    case ErrScore:
        std::sort(ret.begin(), ret.end(), compareErrScore);
        break;
    case RankScore:
        std::sort(ret.begin(), ret.end(), compareRankScore);
        break;
    }

    return ret;
}

void Backlog::wait()
{
    if ( waiting ) {
        execDone.wait();
        waiting = false;
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
                target = currentExperiment->models[et->id].uid;
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
            *out << generation << '\t' << stim << '\t' << tmp.begin()->err[stim] << '\t' << mean.err[0] << '\t' << sd.err[0];
            for ( int i = 0; i < NPARAM; i++ ) {
                *out << '\t' << tmp.begin()->param[i] << '\t' << mean.param[i] << '\t' << sd.param[i];
            }
            *out << endl;

            // Replace outdated log with tmp
            log.clear();
            log.splice(log.begin(), tmp);
        }
        execDone.signal();
    }
}


} // namespace backlog


extern "C" backlog::BacklogVirtual *BacklogCreate(int size, int nstims, ostream *out) {
    return new backlog::Backlog(size, nstims, out);
}

extern "C" void BacklogDestroy(backlog::BacklogVirtual **log) {
    delete *log;
    *log = NULL;
}
