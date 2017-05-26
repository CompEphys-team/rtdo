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

    void makeSelection(const WavegenSelection &selection); //!< Finalises selection and adds it to the selection database
    void makeSubset(WaveSource src, std::vector<size_t> indices); //!< Creates a subset and adds it to the subset database
    bool makeDeck(const std::vector<WaveSource> &src); //!< Creates a deck from parameter-ordered single-wave sources (@see WaveDeck::setSource) and adds it to the decks database
    void makeManual(std::vector<Stimulation> stim); //!< Adds a newly generated random/manual stimulation to the manual database

    inline const std::vector<WavegenSelection> &selections() const { return m_selections; }
    inline const std::vector<WaveSubset> &subsets() const { return m_subsets; }
    inline const std::vector<WaveDeck> &decks() const { return m_decks; }
    inline const std::vector<std::vector<Stimulation>> &manuals() const { return m_manual; }
    std::vector<WaveSource> sources() const; //!< Collects all extant Wavegen::Archives, Selections, Subsets, Decks, manual stimulations into a single vector of Sources.

signals:
    void addedSet(); //!< Notifies addition of any WaveSource, including Wavegen::Archive additions
    void addedSelection();
    void addedSubset();
    void addedDeck();
    void addedManual();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "WavesetCreator"; }

    const static QString actionSelect, actionSubset, actionDeck, actionManual;
    const static quint32 magicSelect, magicSubset, magicDeck, magicManual;
    const static quint32 versionSelect, versionSubset, versionDeck, versionManual;

    std::vector<WavegenSelection> m_selections;
    std::vector<WaveSubset> m_subsets;
    std::vector<WaveDeck> m_decks;
    std::vector<std::vector<Stimulation>> m_manual;
};

#endif // WAVESETCREATOR_H
