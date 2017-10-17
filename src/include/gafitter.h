#ifndef GAFITTER_H
#define GAFITTER_H

#include "sessionworker.h"
#include "experimentlibrary.h"
#include "wavesubsets.h"
#include "queue.h"

class GAFitter : public SessionWorker
{
    Q_OBJECT
public:
    GAFitter(Session &session);
    ~GAFitter();

    void abort();

    ExperimentLibrary &lib;

    const GAFitterSettings &settings;

    RTMaybe::Queue<DataPoint> *qV, *qI, *qO;
    double qT;

    struct Output : public Result
    {
        Output(const GAFitter &fitter, Result r = Result());

        std::vector<std::vector<scalar>> params; //!< Best-performing model's parameters, indexed by [epoch][param]
        std::vector<scalar> error; //!< Best performing model's error in each epoch
        std::vector<quint32> stimIdx; //!< Index of the stimulation at each epoch
        std::vector<scalar> targets; //!< Simulator's parameters
        quint32 epochs;
        WaveSource deck;

        bool final;
        std::vector<scalar> finalParams; //!< Final best-performing model across all stimulations
        std::vector<scalar> finalError; //!< Error of the final model on each stimulation
    };

    inline const std::vector<Output> &results() const { return m_results; }
    Output currentResults() const { return output; }

public slots:
    void run(WaveSource src);
    void finish();

signals:
    void starting();
    void didAbort();
    void done();
    void progress(quint32 epoch);

protected slots:
    void clearAbort();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results, Result r);
    inline QString actorName() const { return "GAFitter"; }

    DAQ *daq;

    bool aborted;
    bool doFinish;

    std::vector<Stimulation> stims;
    quint32 stimIdx;
    quint32 epoch;
    std::vector<double> bias;

    void populate();
    void stimulate(const Stimulation &I);
    void procreate();
    void finalise();
    quint32 findNextStim();
    bool finished();
    void pushToQ(double t, double V, double I, double O);

    struct errTupel {
        size_t idx;
        scalar err;
    };
    static bool errTupelSort(const errTupel &x, const errTupel &y);
    std::vector<errTupel> p_err; //!< Sortable errTupels used in procreate()

    std::vector<Output> m_results;
    Output output;

    const static QString action;
    const static quint32 magic, version;
};

#endif // GAFITTER_H
