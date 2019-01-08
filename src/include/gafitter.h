#ifndef GAFITTER_H
#define GAFITTER_H

#include "sessionworker.h"
#include "universallibrary.h"
#include "wavesubsets.h"
#include "queue.h"
#include "daqfilter.h"

class GAFitter : public SessionWorker
{
    Q_OBJECT
public:
    GAFitter(Session &session);
    ~GAFitter();

    UniversalLibrary &lib;

    const GAFitterSettings &settings;

    RTMaybe::Queue<DataPoint> *qV, *qI, *qO;
    double qT;

    struct Output : public Result
    {
        Output(WaveSource stimSource, QString VCRecord, CannedDAQ::ChannelAssociation assoc);
        Output(const GAFitter &fitter, Result r = Result());

        std::vector<std::vector<scalar>> params; //!< Best-performing model's parameters, indexed by [epoch][param]
        std::vector<scalar> error; //!< Best performing model's error in each epoch
        std::vector<quint32> targetParam; //!< Index of the primary target parameter at each epoch
        std::vector<scalar> targets; //!< Simulator's parameters
        quint32 epochs;
        WaveSource stimSource;
        double variance = 0;

        QString VCRecord;
        CannedDAQ::ChannelAssociation assoc;

        bool final;
        std::vector<scalar> finalParams; //!< Final best-performing model across all stimulations
        std::vector<scalar> finalError; //!< Error of the final model on each stimulation

        QVector<iStimulation> stims; //!< Stimulations used by parameter
        QVector<iObservations> obs; //!< Observations by stim
        QVector<QVector<double>> baseF; //!< Mutation rate by stim
    };

    inline const std::vector<Output> &results() const { return m_results; }
    Output currentResults() const { return output; }

    inline QString actorName() const { return "GAFitter"; }
    bool execute(QString action, QString args, Result *res, QFile &file);

public slots:
    void run(WaveSource src, QString VCRecord = "", bool readRecConfig = false);
    void finish();

signals:
    void starting();
    void done();
    void progress(quint32 epoch);

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);

    DAQFilter *daq;

    bool doFinish;

    quint32 targetParam;
    quint32 epoch;
    std::vector<double> bias;

    void setup();
    double fit();

    double getVariance(Stimulation stim);
    void populate();
    double stimulate(unsigned int extra_assignments = 0);
    void procreate();
    double finalise();
    quint32 findNextStim();
    bool finished();
    void pushToQ(double t, double V, double I, double O);

    void save(QFile &file);

    struct errTupel {
        size_t idx = 0;
        scalar err = 0;
    };
    static bool errTupelSort(const errTupel &x, const errTupel &y);

    std::vector<Output> m_results;
    Output output;

    const static QString action;
    const static quint32 magic, version;



// *************** cluster / DE stuff ************************ //
    std::vector<std::vector<std::vector<Section>>> constructClustersByStim(std::vector<Stimulation> astims);

    void procreateDE();
    std::vector<int> DEMethodUsed, DEMethodSuccess, DEMethodFailed;
    std::vector<double> DEpX;

    std::vector<Stimulation> astims;
    std::vector<iStimulation> stims;
    std::vector<iObservations> obs;
    std::vector<int> errNorm;
    std::vector<std::vector<double>> baseF;
};

#endif // GAFITTER_H
