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


#ifndef WAVESOURCE_H
#define WAVESOURCE_H

#include "wavegenselector.h"

class WaveSubset;
class WaveDeck;

class WaveSource
{
public:
    enum Type {
        Archive = 0,
        Selection = 1,
        Subset = 2,
        Deck = 3,
        Manual = 4,
        Empty = 5
    };

    Type type;
    size_t idx;
    int waveno;
    Session *session;

    WaveSource() : type(Empty), idx(0), waveno(-1), session(nullptr) {}
    WaveSource(Session &session, Type type, size_t idx, int waveno = -1) : type(type), idx(idx), waveno(waveno), session(&session) {}

    const Wavegen::Archive *archive() const;
    const WavegenSelection *selection() const; //!< returns the nearest ancestor Selection, if any
    const WaveSubset *subset() const; //!< returns the nearest ancestor Subset, if any
    const WaveDeck *deck() const; //!< returns the deck, if type is Deck

    int resultIndex() const;

    std::vector<Stimulation> stimulations() const;
    std::vector<MAPElite> elites() const;
    std::vector<iStimulation> iStimulations(double dt = 0) const;
    std::vector<iObservations> observations(double dt = 0) const;

    QString prettyName() const;
    int index() const; //!< Returns the overall index (eg. for comboboxes), ignoring waveno.

    friend QDataStream &operator<<(QDataStream &os, const WaveSource &);
    friend QDataStream &operator>>(QDataStream &is, WaveSource &);
    friend bool operator==(const WaveSource &lhs, const WaveSource &rhs);

    static constexpr quint32 version = 100;
};

Q_DECLARE_METATYPE(WaveSource)

#endif // WAVESOURCE_H
