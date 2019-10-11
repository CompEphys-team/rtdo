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


#ifndef WAVESETCREATOR_H
#define WAVESETCREATOR_H

#include "sessionworker.h"
#include "wavegenselector.h"
#include "wavesubsets.h"

class WavesetCreator : public SessionWorker
{
    Q_OBJECT
public:
    WavesetCreator(Session &session);

    inline const std::vector<WavegenSelection> &selections() const { return m_selections; }
    inline const std::vector<WaveSubset> &subsets() const { return m_subsets; }
    inline const std::vector<WaveDeck> &decks() const { return m_decks; }
    inline const std::vector<ManualWaveset> &manuals() const { return m_manual; }
    std::vector<WaveSource> sources() const; //!< Collects all extant Wavegen::Archives, Selections, Subsets, Decks, manual stimulations into a single vector of Sources.

    inline QString actorName() const { return "WavesetCreator"; }
    bool execute(QString action, QString args, Result *res, QFile &file);
    const static QString actionSelect, actionSubset, actionDeck, actionManual, actionManualDeck;

    // Read/write stimulation (meta-)data, including sampling and oversampling rates
    static void writeStims(std::vector<Stimulation> stims, std::ostream &file, double dt);
    static void readStims(std::vector<Stimulation> &stims, std::istream &file, double &dt);

signals:
    void addedSet(); //!< Notifies addition of any WaveSource, including Wavegen::Archive additions
    void addedSelection();
    void addedSubset();
    void addedDeck();
    void addedManual();

protected:
    friend class Session;
    Result *load(const QString &action, const QString &args, QFile &results, Result r);
    const static quint32 magicSelect, magicSubset, magicDeck, magicManual, magicManualDeck;
    const static quint32 versionSelect, versionSubset, versionDeck, versionManual, versionManualDeck;

    std::vector<WavegenSelection> m_selections;
    std::vector<WaveSubset> m_subsets;
    std::vector<WaveDeck> m_decks;
    std::vector<ManualWaveset> m_manual;
};

#endif // WAVESETCREATOR_H
