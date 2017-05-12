#ifndef GAFITTER_H
#define GAFITTER_H

#include "sessionworker.h"
#include "experimentlibrary.h"
#include "wavesubsets.h"

class GAFitter : public SessionWorker
{
    Q_OBJECT
public:
    GAFitter(Session &session, DAQ *daq = nullptr);
    ~GAFitter();

    void stageDeck(WaveSource deck); //!< Enter a deck to be used on next run()

    void abort();

    ExperimentLibrary &lib;

    const GAFitterSettings &settings;

    struct Output {
        Output(const GAFitter &fitter);

        std::vector<std::vector<scalar>> params; //!< Best-performing model's parameters, indexed by [epoch][param]
        std::vector<scalar> error; //!< Best performing model's error in each epoch
        std::vector<quint32> stimIdx; //!< Index of the stimulation at each epoch
        quint32 epochs;
        WaveSource deck;
    };

    inline const std::vector<Output> &results() const { return m_results; }
    Output currentResults() const { return output; }

public slots:
    void run();

signals:
    void didAbort();
    void done();
    void progress(quint32 epoch);

protected slots:
    void clearAbort();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "GAFitter"; }

    DAQ *simulator;
    DAQ *daq;

    randutils::mt19937_rng RNG;

    bool aborted;

    WaveSource stagedDeck;

    WaveDeck deck;
    quint32 stimIdx;
    quint32 epoch;

    void populate();
    void stimulate(const Stimulation &I);
    void settle(const Stimulation &I);
    void procreate();
    bool finished();

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
