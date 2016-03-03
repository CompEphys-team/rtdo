/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-03-02

--------------------------------------------------------------------------*/
#include "experimentaldata.h"
#include <numeric>
#include <algorithm>
#include "shared.h"
#include "realtimeenvironment.h"
#include "config.h"

#define NUM_SAMPLES 10
#define USE_MEDIAN true

SingleChannelData::SingleChannelData(size_t nstim) :
    ExperimentalData(),
    responses(nstim),
    mean(nstim),
    median(nstim),
    nstim(nstim)
{}

void SingleChannelData::startSample(const inputSpec &wave, int stimulation)
{
    stim = stimulation;
    samp = 0;
    if ( responses.at(stim).size() < NUM_SAMPLES ) {
        sampling = true;
        RealtimeEnvironment::env()->setWaveform(wave);
        RealtimeEnvironment::env()->sync();
        wip = vector<double>(wave.t / config->io.dt, {0});
    } else {
        sampling = false;
    }
}

double SingleChannelData::nextSample(size_t channel)
{
    double ret;
    if ( !sampling ) {
        if ( USE_MEDIAN )
            ret = median.at(stim).at(samp);
        else
            ret = mean.at(stim).at(samp);
    } else {
        wip[samp] = RealtimeEnvironment::env()->nextSample(channel);
        if ( USE_MEDIAN && median.at(stim).size() ) {
            ret = median.at(stim).at(samp);
        } else if ( !USE_MEDIAN && mean.at(stim).size() ) {
            ret = mean.at(stim).at(samp);
        } else {
            ret = wip.at(samp);
        }
    }

    samp++;
    return ret;
}

void SingleChannelData::endSample()
{
    if ( sampling ) {
        RealtimeEnvironment::env()->pause();
        responses[stim].push_back(wip);
        calcAverages();
    }
}

void SingleChannelData::calcAverages()
{
    if ( responses.at(stim).size() > 1 ) {
        vector<double> tmp(responses.at(stim).size());
        auto mid = tmp.begin() + (tmp.size() / 2);
        int nT = responses.at(stim).front().size();
        for ( int t = 0; t < nT; t++ ) {
            int i = 0;
            for ( auto &r : responses.at(stim) ) {
                tmp[i++] = r.at(t);
            }
            nth_element(tmp.begin(), mid, tmp.end());
            median[stim][t] = *mid;
            mean[stim][t] = accumulate(tmp.begin(), tmp.end(), double(0)) / tmp.size();
        }
    } else {
        median[stim] = wip;
        mean[stim] = wip;
    }
}

void SingleChannelData::clear()
{
    for ( size_t i = 0; i < nstim; i++ ) {
        responses[i].clear();
        mean[i].clear();
        median[i].clear();
    }
}


MultiChannelData::MultiChannelData(size_t nstim, size_t nchan) :
    ExperimentalData(),
    channels(nchan, SingleChannelData(nstim))
{}

void MultiChannelData::startSample(const inputSpec &wave, int stimulation)
{
    for ( SingleChannelData &d : channels ) {
        d.startSample(wave, stimulation);
    }
}

void MultiChannelData::endSample()
{
    for ( SingleChannelData &d : channels ) {
        d.endSample();
    }
}

double MultiChannelData::nextSample(size_t channel)
{
    double ret = 0;
    for ( size_t i = 0; i < channels.size(); i++ ) {
        if ( i == channel )
            ret = channels[i].nextSample(i);
        else
            channels[i].nextSample(i);
    }
    return ret;
}

void MultiChannelData::calcAverages()
{
    for ( SingleChannelData &d : channels ) {
        d.calcAverages();
    }
}

void MultiChannelData::clear()
{
    for ( SingleChannelData &d : channels ) {
        d.clear();
    }
}


