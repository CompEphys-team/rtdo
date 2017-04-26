#include "wavesubsets.h"
#include "session.h"

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



WaveDeck::WaveDeck(Session &session) :
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
