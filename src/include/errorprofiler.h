/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#ifndef ERRORPROFILER_H
#define ERRORPROFILER_H

#include "sessionworker.h"
#include "universallibrary.h"
#include "errorprofile.h"


/**
 * @brief The ErrorProfiler class is a tool for grid sampling style sensitivity analysis.
 *
 * In a nutshell, it can vary one or more parameters of a model within a set range,
 * stimulate the perturbed models, and record the error that results. The results
 * are offered as a set of vector-like objects (@see ErrorProfile::Profile), each
 * containing the values seen along one axis of the sampling grid.
 * The idea is to get a sense of how sensitive a certain stimulation is to changes
 * in its target parameter over a wide range of candidate models.
 */
class ErrorProfiler : public SessionWorker
{
    Q_OBJECT

public:
    ErrorProfiler(Session &session);
    ~ErrorProfiler();

    /**
     * @brief queueProfile adds a new ErrorProfile to the queue. Once running,
     * a progress() signal is emitted at the end of each simulation block to allow users to keep track of progress.
     * When the profiling is complete, a done() signal is also emitted. Only then does the resulting ErrorProfile
     * become accessible through profiles().
     */
    void queueProfile(ErrorProfile &&p);

    const inline std::vector<ErrorProfile> &profiles() const { return m_profiles; }

    bool execute(QString action, QString args, Result *res, QFile &file);
    inline QString actorName() const { return "ErrorProfiler"; }

signals:
    void progress(int nth, int total);
    void done();
    void didAbort();

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);

    friend class ErrorProfile;
    void stimulate(const iStimulation &stim, const iObservations &obs);

private:
    UniversalLibrary &lib;
    DAQ *daq;

    std::list<ErrorProfile> m_queue;
    std::vector<ErrorProfile> m_profiles;

    const static QString action;
    const static quint32 magic, version;
};

#endif // ERRORPROFILER_H
