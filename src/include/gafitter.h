#ifndef GAFITTER_H
#define GAFITTER_H

#include "sessionworker.h"
#include "experimentlibrary.h"
#include "wavesubsets.h"
#include "queue.h"
#include "daqfilter.h"

class GAFitter : public SessionWorker
{
    Q_OBJECT
public:
    GAFitter(Session &session);
    ~GAFitter();

    ExperimentLibrary &lib;

    const GAFitterSettings &settings;

    RTMaybe::Queue<DataPoint> *qV, *qI, *qO;
    double qT;

    struct Output : public Result
    {
        Output(WaveSource deck, QString VCRecord, CannedDAQ::ChannelAssociation assoc);
        Output(const GAFitter &fitter, Result r = Result());

        std::vector<std::vector<scalar>> params; //!< Best-performing model's parameters, indexed by [epoch][param]
        std::vector<scalar> error; //!< Best performing model's error in each epoch
        std::vector<quint32> stimIdx; //!< Index of the stimulation at each epoch
        std::vector<scalar> targets; //!< Simulator's parameters
        quint32 epochs;
        WaveSource deck;
        double variance;

        QString VCRecord;
        CannedDAQ::ChannelAssociation assoc; //!< Runtime only: Channel association & scaling

        bool final;
        std::vector<scalar> finalParams; //!< Final best-performing model across all stimulations
        std::vector<scalar> finalError; //!< Error of the final model on each stimulation
    };

    inline const std::vector<Output> &results() const { return m_results; }
    Output currentResults() const { return output; }

    inline QString actorName() const { return "GAFitter"; }
    bool execute(QString action, QString args, Result *res, QFile &file);

    std::vector<Stimulation> sanitiseDeck(std::vector<Stimulation> stimulations, bool useQueuedSettings = false);

public slots:
    void run(WaveSource src, QString VCRecord, CannedDAQ::ChannelAssociation assoc);
    void finish();

signals:
    void starting();
    void done();
    void progress(quint32 epoch);

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results, Result r);

    DAQFilter *daq;

    bool doFinish;

    quint32 stimIdx;
    quint32 epoch;
    std::vector<double> bias;

    void setup(const std::vector<Stimulation> &astims);
    double fit();

    double getVariance(Stimulation stim);
    void populate();
    double stimulate();
    void procreate();
    void finalise();
    quint32 findNextStim();
    bool finished();
    void pushToQ(double t, double V, double I, double O);

    void save(QFile &file);

    struct errTupel {
        size_t idx = 0;
        scalar err = 0;
    };
    static bool errTupelSort(const errTupel &x, const errTupel &y);
    std::vector<errTupel> p_err; //!< Sortable errTupels used in procreate()

    std::vector<Output> m_results;
    Output output;

    const static QString action;
    const static quint32 magic, version;



// *************** cluster / DE stuff ************************ //
    std::vector<std::vector<std::vector<Section>>> constructClustersByStim(std::vector<Stimulation> astims);

    double stimulate_cluster(const std::vector<Section> &cluster, int stimIdx);

    void resetDE();
    void procreateDE();
    std::vector<int> DEMethodUsed, DEMethodSuccess, DEMethodFailed;
    std::vector<double> DEpX;

    std::vector<Stimulation> stims;
    std::vector<std::vector<std::pair<int,int>>> obsTimes;
    std::vector<int> errNorm;
    std::vector<std::vector<double>> baseF;
};

#endif // GAFITTER_H
