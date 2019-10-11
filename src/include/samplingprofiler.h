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


#ifndef SAMPLINGPROFILER_H
#define SAMPLINGPROFILER_H

#include "sessionworker.h"
#include "universallibrary.h"
#include "wavesource.h"

class SamplingProfiler : public SessionWorker
{
    Q_OBJECT
public:
    SamplingProfiler(Session &session);
    ~SamplingProfiler() {}

    struct Profile : public Result
    {
        Profile(Result r = Result()) : Result(r) {}
        Profile(WaveSource src, size_t target, Result r = Result()); //!< Populates sensible defaults for sigma and sizes the vectors.
        WaveSource src; //!< Stimulations to profile.
        size_t target; //!< Parameter to profile.

        QVector<bool> uniform; //!< Flag for each parameter: true to interpret value1/value2 as min/max rather than mean/sd.
        QVector<double> value1, value2; //!< min/max or mean/sd for each parameter, depending on uniform. No distinction is made between additive and multiplicative params.

        // Outputs, each sized src.stimulations().size():
        QVector<double> rho_weighted;       //!< Output: Pearson's correlation coefficient between error and deviation-weighted param space distance
        QVector<double> rho_unweighted;     //!< Output: Correlation between error and euclidean param space distance
        QVector<double> rho_target_only;    //!< Output: Correlation between error and target parameter distance

        // Corresponding gradients (mean err / mean distance)
        QVector<double> grad_weighted;
        QVector<double> grad_unweighted;
        QVector<double> grad_target_only;
    };

    const inline std::vector<Profile> &profiles() const { return m_profiles; }

    bool execute(QString action, QString args, Result *res, QFile &file);
    inline QString actorName() const { return "SamplingProfiler"; }

public slots:
    void generate(SamplingProfiler::Profile);

signals:
    void progress(int nth, int total);
    void done();
    void didAbort();

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);

private:
    UniversalLibrary &lib;

    std::vector<Profile> m_profiles;

    const static QString action;
    const static quint32 magic, version;
};

Q_DECLARE_METATYPE(SamplingProfiler::Profile)

#endif // SAMPLINGPROFILER_H
