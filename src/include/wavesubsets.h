#ifndef WAVESUBSETS_H
#define WAVESUBSETS_H

#include "wavesource.h"

class WaveSubset : public Result
{
public:
    WaveSubset(WaveSource src, std::vector<size_t> indices, Result r = Result()) : Result(r), src(std::move(src)), indices(std::move(indices)) {}
    std::vector<Stimulation> stimulations() const;
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
    const std::vector<Stimulation> &stimulations() const { return m_stimulations; }

private:
    std::vector<WaveSource> src;
    std::vector<Stimulation> m_stimulations;
};

class ManualWaveset : public Result
{
public:
    ManualWaveset(std::vector<Stimulation> stims, Result r = Result()) : Result(r), stims(stims) {}
    std::vector<Stimulation> stims;
};

#endif // WAVESUBSETS_H
