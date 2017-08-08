#ifndef ERRORPROFILER_H
#define ERRORPROFILER_H

#include "sessionworker.h"
#include "experimentlibrary.h"
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

    void abort(); //!< Abort all queued slot actions.

    /**
     * @brief queueProfile adds a new ErrorProfile to the queue for generate() to work with.
     * A typical workflow will not require multiple ErrorProfiles in the queue, but passing them
     * directly to the slot is tricky. To clear the queue, call abort().
     */
    bool queueProfile(ErrorProfile &&p);

    const inline std::vector<ErrorProfile> &profiles() const { return m_profiles; }

public slots:
    /**
     * @brief generate runs the simulations set up in the first queued ErrorProfile.
     * A progress() signal is emitted at the end of each simulation block to allow users to keep track of progress.
     * When the profiling is complete, a done() signal is also emitted. Only then does the resulting ErrorProfile
     * become accessible through profiles().
     */
    void generate();

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
    inline QString actorName() const { return "ErrorProfiler"; }

    friend class ErrorProfile;
    void settle(scalar baseV, scalar settleDuration);
    void stimulate(const Stimulation &stim);

private:
    ExperimentLibrary &lib;
    DAQ *daq;

    bool aborted;

    std::list<ErrorProfile> m_queue;
    std::vector<ErrorProfile> m_profiles;

    const static QString action;
    const static quint32 magic, version;
};

#endif // ERRORPROFILER_H
