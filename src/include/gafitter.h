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
    struct errTupel {
        size_t idx = 0;
        scalar err = 0;
    };
    static bool errTupelSort(const errTupel &x, const errTupel &y);

    GAFitter(Session &session);
    ~GAFitter();

    UniversalLibrary &lib;

    const GAFitterSettings &settings;

    RTMaybe::Queue<DataPoint> *qV, *qI, *qO, *qV2;
    double qT;

    struct Output : public Result
    {
        Output() = default;
        Output(const Output &) = default;
        Output(Output&&) = default;
        Output &operator=(const Output&) = default;
        Output &operator=(Output&&) = default;

        Output(Session &s, WaveSource stimSource, QString VCRecord, Result r = Result());

        std::vector<std::vector<scalar>> params; //!< Best-performing model's parameters, indexed by [epoch][param]
        std::vector<scalar> error; //!< Best performing model's error in each epoch
        std::vector<quint32> targetStim; //!< Index of the stimulation used at each epoch (==target param for decks)
        std::vector<scalar> targets; //!< Simulator's parameters
        quint32 epochs = 0;
        WaveSource stimSource;
        double variance = 0;

        QString VCRecord;
        CannedDAQ::ChannelAssociation assoc;

        bool final = false;
        std::vector<scalar> finalParams; //!< Final best-performing model across all stimulations
        std::vector<scalar> finalError; //!< Error of the final model on each stimulation

        QVector<iStimulation> stims; //!< Stimulations used by parameter
        QVector<iObservations> obs; //!< Observations by stim
        QVector<QVector<double>> baseF; //!< Mutation rate by stim

        // Resumability
        struct {
            std::vector<std::vector<scalar>> population;
            std::vector<double> bias;
            std::vector<double> DEMethodSuccess, DEMethodFailed;
            std::vector<double> DEpX;
        } resume;

        bool closedLoop = false;
    };

    struct Validation : public Result
    {
        Validation(Result r = Result()) : Result(r) {}
        int fitIdx;
        QVector<QVector<double>> error;
        QVector<double> mean, sd;
    };

    inline const std::vector<Output> &results() const { return m_results; }
    Output currentResults() const { QMutexLocker locker(&mutex); if ( running) return output; else return m_results.back(); }

    inline const std::vector<Validation> &validations() const { return m_validations; }

    inline QString actorName() const { return "GAFitter"; }
    bool execute(QString action, QString args, Result *res, QFile &file);

    bool cl_exec(Result *res, QFile &file);

    // Utility
    static std::vector<std::vector<double>> load_validation(QFile &base, int ep);

public slots:
    void run(WaveSource src, QString VCRecord = "", bool readRecConfig = false);
    void resume(size_t fitIdx, WaveSource src, QString VCRecord = "", bool readRecConfig = false);
    void finish();

    void cl_run(WaveSource src);
    void cl_resume(size_t fitIdx, WaveSource src);

    void validate(size_t fitIdx);

signals:
    void starting();
    void done();
    void progress(quint32 epoch);
    void pca_complete();

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);

    DAQFilter *daq;

    bool doFinish;
    bool running = false;

    quint32 targetStim;
    quint32 epoch;
    std::vector<double> bias;

    void setup(bool ad_hoc_stims = false);
    double fit(QFile &file);

    void cl_fit(QFile &file);
    void cl_settle();
    std::vector<iStimulation> cl_findStims(QFile&);
    void cl_stimulate(QFile &file, int stimIdx);
    void cl_pca();

    bool exec_validation(Result *res, QFile &file);
    void save_validation_result(QFile &file);
    Result *load_validation_result(Result r, QFile &file, const QString &args);

    void record_validation(QFile&);

    void populate();

    double stimulate(unsigned int extra_assignments = 0);
    void stimulateMonolithic();
    void stimulateChunked();

    std::vector<errTupel> procreate();
    void procreate_subpopulation(std::vector<errTupel> &p_err, size_t modelOffset, size_t nModels);
    double finalise();
    quint32 findNextStim();
    bool finished();
    void pushToQ(double t, double V, double I, double O);

    void save(QFile &file);

    std::vector<Output> m_results;
    Output output;

    const static QString action, cl_action, validate_action;
    const static quint32 magic, version, validate_magic, validate_version;

    void procreateDE();
    void procreateDE_subpopulation(int modelOffset, int nModels, int offspringOffset, const std::vector<double> &methodCutoff, double successRateTotal, int bestIdx);
    std::vector<int> DEMethodUsed;
    std::vector<double> DEMethodSuccess, DEMethodFailed;
    std::vector<double> DEpXUsed, DEpX;

    std::vector<Stimulation> astims;
    std::vector<iStimulation> stims;
    std::vector<iObservations> obs;
    std::vector<int> errNorm;
    std::vector<std::vector<double>> baseF;

    std::vector<Validation> m_validations;
};

#endif // GAFITTER_H
