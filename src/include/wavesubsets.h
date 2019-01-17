#ifndef WAVESUBSETS_H
#define WAVESUBSETS_H

#include "wavesource.h"

class WaveSubset : public Result
{
public:
    WaveSubset(WaveSource src, std::vector<size_t> indices, Result r = Result()) : Result(r), src(std::move(src)), indices(std::move(indices)) {}
    std::vector<iStimulation> stimulations(double dt) const;
    inline size_t size() const { return indices.size(); }
    QString prettyName() const;

    WaveSource src;
    std::vector<size_t> indices;
};

class WaveDeck : public Result
{
public:
    WaveDeck(Session &session, std::vector<WaveSource> sources);
    WaveDeck(Session &session, Result r = Result());

    bool setSource(size_t targetParam, WaveSource source); //!< Sets the stimulation for the given parameter. source must refer to a single stimulation.
    const std::vector<WaveSource> &sources() const { return src; }
    const std::vector<iStimulation> &stimulations() const { return m_stimulations; }
    const std::vector<iObservations> &obs() const { return m_obs; }

private:
    std::vector<WaveSource> src;
    std::vector<iStimulation> m_stimulations;
    std::vector<iObservations> m_obs;
};

class ManualWaveset : public Result
{
public:
    ManualWaveset(std::vector<iStimulation> stims, std::vector<iObservations> obs, Result r = Result()) : Result(r), stims(stims), observations(obs) {}
    std::vector<iStimulation> stims;
    std::vector<iObservations> observations;
};

#endif // WAVESUBSETS_H
