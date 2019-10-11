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
