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

SingleChannelData::SingleChannelData(size_t nstim) :
    ExperimentalData(),
    responses(nstim),
    mean(nstim),
    median(nstim),
    nstim(nstim),
    lockSampling(false)
{}

void SingleChannelData::startSample(const inputSpec &wave, int stimulation)
{
    stim = stimulation;
    samp = 0;
    if ( !lockSampling && (responses.at(stim).size() < size || !size) ) {
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
        if ( config->vc.cacheUseMedian )
            ret = median.at(stim).at(samp);
        else
            ret = mean.at(stim).at(samp);
    } else {
        wip[samp] = RealtimeEnvironment::env()->nextSample(channel);
        if ( config->vc.cacheUseMedian && median.at(stim).size() ) {
            ret = median.at(stim).at(samp);
        } else if ( !config->vc.cacheUseMedian && mean.at(stim).size() ) {
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
    lockSampling = false;
}

void SingleChannelData::setSize(size_t cacheSize)
{
    if ( cacheSize < size )
        clear();
    size = cacheSize;
    lockSampling = false;
}

void SingleChannelData::dump(ostream &os)
{
    for ( size_t i = 0; i < nstim; i++ ) {
        if ( !mean.at(i).size() || !median.at(i).size() ) {
            cerr << "Skipping empty dataset at index " << i << endl;
            continue;
        } else if ( (responses.at(i).size() && responses.at(i).front().size() != mean.at(i).size())
                    || mean.at(i).size() != median.at(i).size() ) {
            cerr << "Skipping incongruously sized responses at index " << i << endl;
            continue;
        }
        os << "# Stimulation " << i << endl;
        os << "mean\tmedian";
        for ( size_t j = 0; j < responses.at(i).size(); j++ ) {
            os << "\t" << j;
        }
        os << endl;
        for ( size_t k = 0; k < mean.at(i).size(); k++ ) {
            os << mean[i][k] << '\t' << median[i][k];
            for ( auto &v : responses.at(i) ) {
                os << '\t' << v[k];
            }
            os << endl;
        }
        os << endl << endl;
    }
}

void SingleChannelData::load(istream &is)
{
    clear();
    char buffer[1024];
    unsigned int ns = 0;
    double tmp;
    while ( is.good() ) {
        while ( is.good() && (is.peek() == '#' || is.peek() == '\n') ) {
            is.getline(buffer, 1024);
        }
        sscanf(buffer, "# Stimulation %u", &ns);
        if ( ns > mean.size() ) {
            continue;
        }
        is.getline(buffer, 1024); // Skip header
        while ( is.good() && is.peek() != '\n' ) {
            is >> tmp;
            mean[ns].push_back(tmp);
            is >> tmp;
            median[ns].push_back(tmp);
            is.getline(buffer, 1024); // Skip individual responses
        }
    }
    lockSampling = true;
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

void MultiChannelData::setSize(size_t cacheSize)
{
    for ( SingleChannelData &d : channels ) {
        d.setSize(cacheSize);
    }
    size = cacheSize;
}

