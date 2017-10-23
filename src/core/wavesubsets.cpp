#include "wavesubsets.h"
#include "session.h"
#include <cassert>

std::vector<Stimulation> WaveSubset::stimulations() const
{
    std::vector<Stimulation> ret, source = src.stimulations();
    ret.reserve(indices.size());
    for ( size_t i : indices )
        ret.push_back(source[i]);
    return ret;
}

QString WaveSubset::prettyName() const
{
    return QString("%1 waves (from %2)").arg(size()).arg(src.prettyName());
}



WaveDeck::WaveDeck(Session &session, std::vector<WaveSource> sources) :
    src(sources),
    m_stimulations(session.project.model().adjustableParams.size())
{
    assert(sources.size() == m_stimulations.size());
    int i = 0;
    for ( const WaveSource &s : src ) {
        std::vector<Stimulation> stim = s.stimulations();
        assert(stim.size() == 1);
        m_stimulations[i++] = std::move(stim[0]);
    }
}

WaveDeck::WaveDeck(Session &session, Result r) :
    Result(r),
    src(session.project.model().adjustableParams.size()),
    m_stimulations(session.project.model().adjustableParams.size())
{

}

bool WaveDeck::setSource(size_t targetParam, WaveSource source)
{
    std::vector<Stimulation> stim = source.stimulations();
    if ( stim.size() != 1 )
        return false;
    m_stimulations[targetParam] = std::move(stim[0]);
    src[targetParam] = std::move(source);
    return true;
}
