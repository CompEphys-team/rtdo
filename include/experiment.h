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

class Experiment
{
public:
    Experiment(conf::Config *cfg, size_t channel) :
        cfg(cfg),
        epoch(0),
        nextS(0),
        channel(channel)
    {}

    virtual ~Experiment() {}

    virtual void initModel() = 0;
    virtual void run(bool *stopFlag) = 0;

    inline shared_ptr<backlog::Backlog> log() const { return logger; }
    inline shared_ptr<ExperimentalData> data() const { return _data; }

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
