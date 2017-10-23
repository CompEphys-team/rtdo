#include "wavesetcreator.h"
#include "session.h"

const QString WavesetCreator::actionSelect = QString("select");
const quint32 WavesetCreator::magicSelect = 0xa54f3955;
const quint32 WavesetCreator::versionSelect = 101;

const QString WavesetCreator::actionSubset = QString("subset");
const quint32 WavesetCreator::magicSubset = 0x12be3df0;
const quint32 WavesetCreator::versionSubset = 100;

const QString WavesetCreator::actionDeck = QString("deck");
const quint32 WavesetCreator::magicDeck = 0xbfb1aa06;
const quint32 WavesetCreator::versionDeck = 100;

const QString WavesetCreator::actionManual = QString("manual");
const quint32 WavesetCreator::magicManual = 0xee4ea782;
const quint32 WavesetCreator::versionManual = 100;

const QString WavesetCreator::actionManualDeck = QString("mdeck");
const quint32 WavesetCreator::magicManualDeck = 0xc8e5a57c;
const quint32 WavesetCreator::versionManualDeck = 100;

WavesetCreator::WavesetCreator(Session &session) :
    SessionWorker(session)
{
    connect(&session.wavegen(), SIGNAL(done(int)), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedSelection()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedSubset()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedDeck()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedManual()), this, SIGNAL(addedSet()));
}

bool WavesetCreator::execute(QString action, QString, Result *res, QFile &file)
{
    QDataStream os;

    if ( action == actionSelect ) {
        m_selections.push_back(*static_cast<WavegenSelection*>(res));
        WavegenSelection &sel = m_selections.back();
        sel.finalise();

        if ( !openSaveStream(file, os, magicSelect, versionSelect) )
            return false;
        // Save archive index in reverse to prevent conflicts during session merge
        os << quint32(session.wavegen().archives().size() - sel.archive_idx);
        os << quint32(sel.ranges.size());
        for ( WavegenSelection::Range const& r : sel.ranges )
            os << quint32(r.min) << quint32(r.max) << r.collapse;
        os << sel.minFitness;

        emit addedSelection();
    } else if ( action == actionSubset ) {
        m_subsets.push_back(*static_cast<WaveSubset*>(res));
        WaveSubset &set = m_subsets.back();

        if ( !openSaveStream(file, os, magicSubset, versionSubset) )
            return false;
        os << set.src << quint32(set.size());
        for ( quint32 i : set.indices )
            os << i;

        emit addedSubset();
    } else if ( action == actionDeck ) {
        m_decks.push_back(*static_cast<WaveDeck*>(res));

        if ( !openSaveStream(file, os, magicDeck, versionDeck) )
            return false;
        for ( const WaveSource &source : m_decks.back().sources() )
            os << source;

        emit addedDeck();
    } else if ( action == actionManual ) {
        m_manual.push_back(*static_cast<ManualWaveset*>(res));

        if ( !openSaveStream(file, os, magicManual, versionManual) )
            return false;
        os << quint32(m_manual.back().stims.size());
        for ( const Stimulation &s : m_manual.back().stims )
            os << s;

        emit addedManual();
    } else if ( action == actionManualDeck ) {
        // Manual waveset saved as deck: Add both manual and deck entries
        m_manual.push_back(*static_cast<ManualWaveset*>(res));
        m_decks.emplace_back(session, *res);
        size_t idx = m_manual.size()-1;
        for ( size_t i = 0; i < session.project.model().adjustableParams.size(); i++ ) {
            m_decks.back().setSource(i, WaveSource(session, WaveSource::Manual, idx, i));
        }

        if ( !openSaveStream(file, os, magicManualDeck, versionManualDeck) )
            return false;
        os << quint32(m_manual.back().stims.size());
        for ( const Stimulation &s : m_manual.back().stims )
            os << s;

        emit addedDeck();
        emit addedManual();
    } else {
        delete res;
        return false;
    }

    delete res;
    return true;
}

std::vector<WaveSource> WavesetCreator::sources() const
{
    // Include, in this order, Wavegen::Archives, Selections, Subsets, Decks, Manuals
    std::vector<WaveSource> src;
    size_t nArchives = session.wavegen().archives().size();
    src.reserve(nArchives + m_selections.size() + m_subsets.size() + m_decks.size());
    for ( size_t i = 0; i < nArchives; i++ ) {
        src.emplace_back(session, WaveSource::Archive, i);
    }
    for ( size_t i = 0; i < m_selections.size(); i++ ) {
        src.emplace_back(session, WaveSource::Selection, i);
    }
    for ( size_t i = 0; i < m_subsets.size(); i++ ) {
        src.emplace_back(session, WaveSource::Subset, i);
    }
    for ( size_t i = 0; i < m_decks.size(); i++ ) {
        src.emplace_back(session, WaveSource::Deck, i);
    }
    for ( size_t i = 0; i < m_manual.size(); i++ ) {
        src.emplace_back(session, WaveSource::Manual, i);
    }
    return src;
}

void WavesetCreator::load(const QString &action, const QString &, QFile &results, Result r)
{
    QDataStream is;
    quint32 version;
    if ( action == actionSelect ) {
        version = openLoadStream(results, is, magicSelect);
        if ( version < 100 || version > versionSelect )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        quint32 idx, nranges, min, max;
        is >> idx >> nranges;
        WavegenSelection sel(session, session.wavegen().archives().size() - idx, r);
        sel.ranges.resize(nranges);
        for ( WavegenSelection::Range &r : sel.ranges ) {
            is >> min >> max >> r.collapse;
            r.min = min;
            r.max = max;
        }
        is >> sel.minFitness;
        sel.finalise();
        m_selections.push_back(std::move(sel));
    } else if ( action == actionSubset ) {
        version = openLoadStream(results, is, magicSubset);
        if ( version < 100 || version > versionSubset)
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        quint32 size, idx;
        m_subsets.push_back(WaveSubset(WaveSource(), {}, r));
        WaveSubset &subset = m_subsets.back();
        subset.src.session =& session;
        is >> subset.src >> size;
        subset.indices.resize(size);
        for ( size_t &i : subset.indices ) {
            is >> idx;
            i = idx;
        }
    } else if ( action == actionDeck ) {
        version = openLoadStream(results, is, magicDeck);
        if ( version < 100 || version > versionDeck )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        WaveSource src;
        src.session =& session;
        WaveDeck deck(session, r);
        for ( size_t i = 0, end = session.project.model().adjustableParams.size(); i < end; i++ ) {
            is >> src;
            deck.setSource(i, src);
        }
        m_decks.push_back(std::move(deck));
    } else if ( action == actionManual ) {
        version = openLoadStream(results, is, magicManual);
        if ( version < 100 || version > versionManual )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        quint32 size;
        is >> size;
        std::vector<Stimulation> stim(size);
        for ( Stimulation &s : stim )
            is >> s;
        m_manual.push_back(ManualWaveset(std::move(stim), r));
    } else if ( action == actionManualDeck ) {
        version = openLoadStream(results, is, magicManualDeck);
        if ( version < 100 || version > versionManualDeck )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        quint32 size;
        is >> size;
        std::vector<Stimulation> stim(size);
        for ( Stimulation &s : stim )
            is >> s;
        m_manual.push_back(ManualWaveset(std::move(stim), r));

        m_decks.emplace_back(session, r);
        size_t idx = m_manual.size()-1;
        for ( size_t i = 0; i < session.project.model().adjustableParams.size(); i++ ) {
            m_decks.back().setSource(i, WaveSource(session, WaveSource::Manual, idx, i));
        }
    } else {
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    }
}
