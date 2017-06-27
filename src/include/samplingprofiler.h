#ifndef SAMPLINGPROFILER_H
#define SAMPLINGPROFILER_H

#include "sessionworker.h"
#include "profilerlibrary.h"
#include "randutils.hpp"
#include "wavesource.h"

class SamplingProfiler : public SessionWorker
{
    Q_OBJECT
public:
    SamplingProfiler(Session &session);
    ~SamplingProfiler() {}

    struct Profile
    {
        Profile() {}
        Profile(WaveSource src); //!< Populates sensible defaults for target and sigma, and sizes the vectors.
        WaveSource src; //!< Stimulations to profile.
        size_t target; //!< Parameter to profile.
        double sigma; //!< Detuning coefficient
        size_t samplingInterval; //!< in cycles; frequency of current evaluation. Defaults to RunData::simCycles.

        QVector<bool> uniform; //!< Flag for each parameter: true to interpret value1/value2 as min/max rather than mean/sd.
        QVector<double> value1, value2; //!< min/max or mean/sd for each parameter, depending on uniform. No distinction is made between additive and multiplicative params.

        // Outputs, each sized src.stimulations().size():
        QVector<double> gradient; //!< Output: Normalised median error gradient for each stimulation
        QVector<double> accuracy; //!< Output: Fraction of gradients correctly indicating target direction
    };

    void abort();

    const inline std::vector<Profile> &profiles() const { return m_profiles; }

public slots:
    void generate(SamplingProfiler::Profile);

signals:
    void progress(int nth, int total);
    void done();
    void doAbort();
    void didAbort();

protected slots:
    void clearAbort();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "SamplingProfiler"; }

private:
    ProfilerLibrary &lib;

    bool aborted;

    std::vector<Profile> m_profiles;

    const static QString action;
    const static quint32 magic, version;

    randutils::mt19937_rng RNG;
};

Q_DECLARE_METATYPE(SamplingProfiler::Profile)

#endif // SAMPLINGPROFILER_H