#include "wavesetcreator.h"
#include "session.h"

const QString WavesetCreator::actionSelect = QString("select");
const quint32 WavesetCreator::magicSelect = 0xa54f3955;
const quint32 WavesetCreator::versionSelect = 100;

const QString WavesetCreator::actionSubset = QString("subset");
const quint32 WavesetCreator::magicSubset = 0x12be3df0;
const quint32 WavesetCreator::versionSubset = 100;

const QString WavesetCreator::actionDeck = QString("deck");
const quint32 WavesetCreator::magicDeck = 0xbfb1aa06;
const quint32 WavesetCreator::versionDeck = 100;

WavesetCreator::WavesetCreator(Session &session) :
    SessionWorker(session)
{
    connect(&session.wavegen(), SIGNAL(done(int)), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedSelection()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedSubset()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedDeck()), this, SIGNAL(addedSet()));
}

void WavesetCreator::select(const WavegenSelection &selection)
{
    m_selections.push_back(selection);
    WavegenSelection &sel = m_selections.back();
    sel.finalise();

    emit addedSelection();

    QFile file(session.log(this, actionSelect));
    QDataStream os;
    if ( !openSaveStream(file, os, magicSelect, versionSelect) )
        return;

    // Save archive index in reverse to prevent conflicts during session merge
    os << quint32(session.wavegen().archives().size() - sel.archive_idx);
    os << quint32(sel.ranges.size());
    for ( WavegenSelection::Range const& r : sel.ranges )
        os << quint32(r.min) << quint32(r.max) << r.collapse;
}

void WavesetCreator::subset(WaveSource src, std::vector<size_t> indices)
{
    m_subsets.push_back(WaveSubset(std::move(src), std::move(indices)));
    WaveSubset &set = m_subsets.back();

    emit addedSubset();

    QFile file(session.log(this, actionSubset));
    QDataStream os;
    if ( !openSaveStream(file, os, magicSubset, versionSubset) )
        return;

    os << set.src << quint32(set.size());
    for ( quint32 i : set.indices )
        os << i;
}

bool WavesetCreator::deck(const std::vector<WaveSource> &src)
{
    if ( src.size() != session.project.model().adjustableParams.size() )
        return false;
    WaveDeck deck(session);
    for ( size_t i = 0; i < src.size(); i++ ) {
        if ( !deck.setSource(i, src[i]) ) {
            return false;
        }
    }
    m_decks.push_back(std::move(deck));

    emit addedDeck();

    QFile file(session.log(this, actionDeck));
    QDataStream os;
    if ( !openSaveStream(file, os, magicDeck, versionDeck) )
        return true;

    for ( const WaveSource &source : src )
        os << source;
    return true;
}

std::vector<WaveSource> WavesetCreator::sources() const
{
    // Include, in this order, Wavegen::Archives, Selections, Subsets, Decks
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
    return src;
}

void WavesetCreator::load(const QString &action, const QString &, QFile &results)
{
    QDataStream is;
    quint32 version;
    if ( action == actionSelect ) {
        version = openLoadStream(results, is, magicSelect);
        if ( version < 100 || version > versionSelect )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        quint32 idx, nranges, min, max;
        is >> idx >> nranges;
        WavegenSelection sel(session, session.wavegen().archives().size() - idx);
        sel.ranges.resize(nranges);
        for ( WavegenSelection::Range &r : sel.ranges ) {
            is >> min >> max >> r.collapse;
            r.min = min;
            r.max = max;
        }
        sel.finalise();
        m_selections.push_back(std::move(sel));
    } else if ( action == actionSubset ) {
        version = openLoadStream(results, is, magicSubset);
        if ( version < 100 || version > versionSubset)
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        quint32 size, idx;
        WaveSource src;
        std::vector<size_t> indices;
        src.session =& session;
        is >> src >> size;
        indices.resize(size);
        for ( size_t &i : indices ) {
            is >> idx;
            i = idx;
        }
        m_subsets.push_back(WaveSubset(std::move(src), std::move(indices)));
    } else if ( action == actionDeck ) {
        version = openLoadStream(results, is, magicDeck);
        if ( version < 100 || version > versionDeck )
            throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

        WaveSource src;
        src.session =& session;
        WaveDeck deck(session);
        for ( size_t i = 0, end = session.project.model().adjustableParams.size(); i < end; i++ ) {
            is >> src;
            deck.setSource(i, src);
        }
        m_decks.push_back(std::move(deck));
    } else {
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    }
}
