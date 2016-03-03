/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-03-02

--------------------------------------------------------------------------*/
#ifndef EXPERIMENTALDATA_H
#define EXPERIMENTALDATA_H

#include <vector>
#include <list>
#include "shared.h"

using namespace std;

class ExperimentalData
{
public:
    ExperimentalData() {}

    virtual void startSample(const inputSpec &wave, int stimulation) = 0;
    virtual void endSample() = 0;
    virtual double nextSample(size_t channel = 0) = 0;

    virtual void calcAverages() = 0;
    virtual void clear() = 0;
};


class SingleChannelData : public ExperimentalData
{
public:
    SingleChannelData(size_t nstim);

    void startSample(const inputSpec &wave, int stimulation);
    void endSample();

    //!< Reads a sample from @arg channel and advances the sample counter. Caller beware of changing channels.
    //! If no previous episodes exist, this returns the live value. Otherwise, this returns the mean/median of PREVIOUS
    //! episodes, i.e. STALE data while the episode quota remains unfulfilled. The mean & median values are updated
    //! only upon calling @fn endSample() at the end of an episode.
    double nextSample(size_t channel = 0);

    void calcAverages();
    void clear();

    //!< A vector of stimulations; a list of episodes; and finally, a vector of the samples of an episode
    //! In other words, a sample is responses[stimulation][episode][time].
    vector<list<vector<double>>> responses;
    vector<vector<double>> mean;
    vector<vector<double>> median;

private:
    size_t nstim;

    vector<double> wip;
    int samp;
    int stim;
    bool sampling;
};


//!< Convenience class that holds a number of SingleChannelData objects to which function calls are redirected.
class MultiChannelData : public ExperimentalData
{
public:
    MultiChannelData(size_t nstim, size_t nchan);

    void startSample(const inputSpec &wave, int stimulation);
    void endSample();

    //!< Note: This call causes sampling from ALL channels, but only returns the value sampled from @arg channel.
    double nextSample(size_t channel = 0);

    void calcAverages();
    void clear();

    vector<SingleChannelData> channels;
};

#endif // EXPERIMENTALDATA_H
