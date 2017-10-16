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

WavesetCreator::WavesetCreator(Session &session) :
    SessionWorker(session)
{
    connect(&session.wavegen(), SIGNAL(done(int)), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedSelection()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedSubset()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedDeck()), this, SIGNAL(addedSet()));
    connect(this, SIGNAL(addedManual()), this, SIGNAL(addedSet()));
}

void WavesetCreator::makeSelection(const WavegenSelection &selection)
{
    m_selections.push_back(selection);
    WavegenSelection &sel = m_selections.back();
    sel.finalise();

    emit addedSelection();

    QFile file(session.log(this, actionSelect, sel));
    QDataStream os;
    if ( !openSaveStream(file, os, magicSelect, versionSelect) )
        return;

    // Save archive index in reverse to prevent conflicts during session merge
    os << quint32(session.wavegen().archives().size() - sel.archive_idx);
    os << quint32(sel.ranges.size());
    for ( WavegenSelection::Range const& r : sel.ranges )
        os << quint32(r.min) << quint32(r.max) << r.collapse;
    os << sel.minFitness;
}

void WavesetCreator::makeSubset(WaveSource src, std::vector<size_t> indices)
{
    m_subsets.push_back(WaveSubset(std::move(src), std::move(indices)));
    WaveSubset &set = m_subsets.back();

    emit addedSubset();

    QFile file(session.log(this, actionSubset, set));
    QDataStream os;
    if ( !openSaveStream(file, os, magicSubset, versionSubset) )
        return;

    os << set.src << quint32(set.size());
    for ( quint32 i : set.indices )
        os << i;
}

bool WavesetCreator::makeDeck(const std::vector<WaveSource> &src)
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

    QFile file(session.log(this, actionDeck, m_decks.back()));
    QDataStream os;
    if ( !openSaveStream(file, os, magicDeck, versionDeck) )
        return true;

    for ( const WaveSource &source : src )
        os << source;
    return true;
}

void WavesetCreator::makeManual(std::vector<Stimulation> stim)
{
    m_manual.push_back(ManualWaveset(stim));
    emit addedManual();

    QFile file(session.log(this, actionManual, m_manual.back()));
    QDataStream os;
    if ( !openSaveStream(file, os, magicManual, versionManual) )
        return;
    os << quint32(stim.size());
    for ( const Stimulation &s : stim )
        os << s;
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
    } else {
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    }
}
