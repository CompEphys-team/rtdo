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

signals:
    void addedSet(); //!< Notifies addition of any WaveSource, including Wavegen::Archive additions
    void addedSelection();
    void addedSubset();
    void addedDeck();
    void addedManual();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results, Result r);
    const static quint32 magicSelect, magicSubset, magicDeck, magicManual, magicManualDeck;
    const static quint32 versionSelect, versionSubset, versionDeck, versionManual, versionManualDeck;

    std::vector<WavegenSelection> m_selections;
    std::vector<WaveSubset> m_subsets;
    std::vector<WaveDeck> m_decks;
    std::vector<ManualWaveset> m_manual;
};

#endif // WAVESETCREATOR_H
