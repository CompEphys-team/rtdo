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

    void stageDeck(WaveSource deck); //!< Enter a deck to be used on next run()

    void abort();

    ExperimentLibrary &lib;

    const GAFitterSettings &settings;

    RTMaybe::Queue<DataPoint> *qV, *qI, *qO;
    double qT;

    struct Output {
        Output(const GAFitter &fitter);

        std::vector<std::vector<scalar>> params; //!< Best-performing model's parameters, indexed by [epoch][param]
        std::vector<scalar> error; //!< Best performing model's error in each epoch
        std::vector<quint32> stimIdx; //!< Index of the stimulation at each epoch
        std::vector<scalar> targets; //!< Simulator's parameters
        quint32 epochs;
        WaveSource deck;
        GAFitterSettings settings;
    };

    inline const std::vector<Output> &results() const { return m_results; }
    Output currentResults() const { return output; }

public slots:
    void run(WaveSource src);

signals:
    void starting();
    void didAbort();
    void done();
    void progress(quint32 epoch);

protected slots:
    void clearAbort();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "GAFitter"; }

    DAQ *daq;

    randutils::mt19937_rng RNG;

    bool aborted;

    WaveDeck deck;
    quint32 stimIdx;
    quint32 epoch;
    std::vector<double> bias;

    void populate();
    void stimulate(const Stimulation &I);
    void settle(const Stimulation &I);
    void procreate();
    quint32 findNextStim();
    bool finished();
    void pushToQ(double tOffset, double V, double I, double O);

    struct errTupel {
        size_t idx;
        scalar err;
    };
    std::vector<errTupel> p_err; //!< Sortable errTupels used in procreate()

    std::vector<Output> m_results;
    Output output;

    const static QString action;
    const static quint32 magic, version;
};

#endif // GAFITTER_H
