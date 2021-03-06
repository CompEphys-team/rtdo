/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-16

--------------------------------------------------------------------------*/
#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "config.h"
#include "backlog.h"
#include "experimentaldata.h"
#include <memory>

class Model
{
public:
    int idx;
    long long uid;
    int parentIdx;
    unsigned long long parentUid;
    double errDiff;
    vector<double> momentum;

    static unsigned long long latestUid;

    Model(int idx);

    void diff();
    void copy(int parentIdx);
    void reinit(int stim);
    void mutate(int stim, double fac, bool retainMomentum);
};

class Experiment
{
friend class Model;
public:
    Experiment(conf::Config *cfg, size_t channel) :
        stopFlag(false),
        cfg(cfg),
        epoch(0),
        nextS(0),
        channel(channel)
    {}

    virtual ~Experiment() {}

    static void *openLibrary();
    static void closeLibrary(void *lib);
    static Experiment *create(void *lib);
    static void destroy(void *lib, Experiment **exp);

    //!< Initialises all models with random values and, if necessary, sets up the runtime environment. Resets the epoch and stimulation counters.
    virtual void initModel() = 0;

    //!< Runs the genetic algorithm until the bool pointed to by @arg stopFlag is true, or other halting conditions are met
    virtual void run(int nEpochs = 0) = 0;

    //!< Applies each stimulation once. Useful for clamp tuning and for final model validation. Use @arg fit to turn model fitting on/off.
    virtual void cycle(bool fit) = 0;

    //!< Evaluate the model indicated by @arg idx for each stimulation, returning its response traces
    virtual vector<vector<double>> stimulateModel(int idx) = 0;

    //!< Inject a parameter set into the live model database at the indicated index, overwriting any previous tenant.
    //! Does not update GA characteristics. Use with caution.
    virtual void injectModel(std::vector<double> params, int idx) = 0;

    //!< Set a parameter to the given value across all models, and prevent further perturbation. Not currently reversible.
    virtual void fixParameter(int paramIdx, double value) = 0;

    //!< Run a CPU simulation of the model at the indicated index, interpreting @arg I as a current clamp stimulus
    virtual vector<double> getCCVoltageTrace(inputSpec I, int idx) = 0;

    virtual void setLog(ostream *out, string closingMessage = string()) = 0;

    inline shared_ptr<backlog::Backlog> log() const { return logger; }
    inline shared_ptr<ExperimentalData> data() const { return _data; }

    vector<Model> models;

    bool stopFlag;

protected:
    conf::Config *cfg;
    shared_ptr<backlog::Backlog> logger;
    shared_ptr<ExperimentalData> _data;

    vector<vector<double>> pperturb;
    vector<vector<double>> sigadjust;
    vector<inputSpec> stims;

    vector<vector<double>> errbuf;
    vector<double> eb;
    vector<double> mavg;
    vector<int> epos;
    vector<int> initial;

    int epoch;
    int nextS;

    int channel;

    void procreateGeneric();
};

#endif // EXPERIMENT_H

