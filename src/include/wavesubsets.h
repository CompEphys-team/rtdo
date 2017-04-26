#ifndef WAVESUBSETS_H
#define WAVESUBSETS_H

#include "wavesource.h"

class WaveSubset
{
public:
    WaveSubset(WaveSource src, std::vector<size_t> indices) : src(std::move(src)), indices(std::move(indices)) {}
    std::vector<Stimulation> stimulations() const;
    inline size_t size() const { return indices.size(); }
    QString prettyName() const;

    WaveSource src;
    std::vector<size_t> indices;
};

class WaveDeck
{
public:
    WaveDeck(Session &session);

    bool setSource(size_t targetParam, WaveSource source); //!< Sets the stimulation for the given parameter. source must refer to a single stimulation.
    const std::vector<WaveSource> &sources() const { return src; }
    const std::vector<Stimulation> &stimulations() const { return m_stimulations; }

private:
    std::vector<WaveSource> src;
    std::vector<Stimulation> m_stimulations;
};

#endif // WAVESUBSETS_H
