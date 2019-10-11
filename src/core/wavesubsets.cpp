/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "wavesubsets.h"
#include "session.h"
#include <cassert>

std::vector<iStimulation> WaveSubset::stimulations(double dt) const
{
    std::vector<iStimulation> ret, source = src.iStimulations(dt);
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
    m_stimulations(session.project.model().adjustableParams.size()),
    m_obs(session.project.model().adjustableParams.size())
{
    assert(sources.size() == m_stimulations.size());
    int i = 0;
    double dt = session.qRunData().dt;
    for ( const WaveSource &s : src ) {
        std::vector<iStimulation> stim = s.iStimulations(dt);
        std::vector<iObservations> obs = s.observations(dt);
        assert(stim.size() == 1);
        m_stimulations[i] = std::move(stim[0]);
        m_obs[i++] = std::move(obs[0]);
    }
}

WaveDeck::WaveDeck(Session &session, Result r) :
    Result(r),
    src(session.project.model().adjustableParams.size()),
    m_stimulations(session.project.model().adjustableParams.size()),
    m_obs(session.project.model().adjustableParams.size())
{

}

bool WaveDeck::setSource(size_t targetParam, WaveSource source)
{
    double dt = source.session->qRunData().dt;
    std::vector<iStimulation> stim = source.iStimulations(dt);
    std::vector<iObservations> obs = source.observations(dt);
    if ( stim.size() != 1 )
        return false;
    m_stimulations[targetParam] = std::move(stim[0]);
    m_obs[targetParam] = std::move(obs[0]);
    src[targetParam] = std::move(source);
    return true;
}
