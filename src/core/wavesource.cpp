#include "wavesource.h"
#include "session.h"

const Wavegen::Archive *WaveSource::archive() const
{
    switch ( type ) {
    default:
    case Archive:   return &session->wavegen().archives().at(idx);
    case Selection: return &selection()->archive();
    case Subset:    return subset()->src.archive();
    case Deck:      return nullptr;
    case Manual:    return nullptr;
    }
}

const WavegenSelection *WaveSource::selection() const
{
    switch ( type ) {
    default:
    case Archive:   return nullptr;
    case Selection: return &session->wavesets().selections().at(idx);
    case Subset:    return subset()->src.selection();
    case Deck:      return nullptr;
    case Manual:    return nullptr;
    }
}

const WaveSubset *WaveSource::subset() const
{
    switch ( type ) {
    default:
    case Archive:
    case Selection: return nullptr;
    case Subset:    return &session->wavesets().subsets().at(idx);
    case Deck:      return nullptr;
    case Manual:    return nullptr;
    }
}

const WaveDeck *WaveSource::deck() const
{
    switch ( type ) {
    default:
    case Archive:
    case Selection:
    case Subset:
    case Manual:
        return nullptr;
    case Deck:
        return &session->wavesets().decks().at(idx);
    }
}

int WaveSource::resultIndex() const
{
    switch ( type ) {
    default:
    case Archive:   return archive()->resultIndex;
    case Selection: return selection()->resultIndex;
    case Subset:    return subset()->resultIndex;
    case Deck:      return deck()->resultIndex;
    case Manual:    return session->wavesets().manuals().at(idx).resultIndex;
    }
}

std::vector<Stimulation> WaveSource::stimulations() const
{
    std::vector<Stimulation> ret;
    bool shrunk = false;
    switch ( type ) {
    case Archive :
    {
        ret.reserve(archive()->elites.size());
        for ( MAPElite const& e : archive()->elites )
            ret.push_back(Stimulation(*e.wave, session->runData(resultIndex()).dt));
        break;
    }
    case Selection :
    {
        std::vector<MAPElite> el = elites();
        shrunk = true;
        ret.reserve(el.size());
        for ( const MAPElite &e : el )
            ret.push_back(Stimulation(*e.wave, session->runData(archive()->resultIndex).dt));
        break;
    }
    case Subset :
        ret = subset()->stimulations();
        break;
    case Deck:
        ret = deck()->stimulations();
        break;
    case Manual:
        ret = session->wavesets().manuals().at(idx).stims;
        break;
    }

    if ( !shrunk && waveno >= 0 )
        return {ret[waveno]};
    else
        return ret;
}

std::vector<MAPElite> WaveSource::elites() const
{
    std::vector<MAPElite> ret;
    switch ( type ) {
    case Archive:
    {
        ret.reserve(archive()->elites.size());
        for ( const MAPElite &el : archive()->elites )
            ret.push_back(el);
        break;
    }
    case Selection:
    {
        const WavegenSelection &sel = *selection();
        ret.reserve(sel.size());
        for ( auto const &it : sel.selection )
            if ( it != nullptr )
                ret.push_back(*it);
        break;
    }
    case Subset:
    {
        std::vector<MAPElite> srcEl = subset()->src.elites();
        ret.reserve(subset()->size());
        for ( size_t i : subset()->indices )
            ret.push_back(srcEl[i]);
        break;
    }
    case Deck:
    {
        ret.reserve(deck()->sources().size());
        for ( const WaveSource &src : deck()->sources() )
            ret.push_back(src.elites()[0]);
        break;
    }
    case Manual:
    {
        std::vector<Stimulation> stims = stimulations();
        ret.resize(stims.size());
        for ( size_t i = 0; i < ret.size(); i++ ) {
            ret[i].wave.reset(new iStimulation(stims[i], session->runData(resultIndex()).dt));
            ret[i].obs.start[0] = ret[i].wave->tObsBegin;
            ret[i].obs.stop[0] = ret[i].wave->tObsEnd;
        }
        break;
    }
    }

    if ( waveno >= 0 )
        return {ret[waveno]};
    else
        return ret;
}

std::vector<iStimulation> WaveSource::iStimulations(double dt) const
{
    std::vector<iStimulation> ret;
    bool shrunk = false;
    switch ( type ) {
    case Archive :
    {
        ret.reserve(archive()->elites.size());
        if ( session->runData(resultIndex()).dt == dt ) {
            for ( MAPElite const& e : archive()->elites )
                ret.push_back(*e.wave);
        } else {
            for ( Stimulation const& I : stimulations() )
                ret.push_back(iStimulation(I, dt));
        }
        break;
    }
    case Selection :
    {
        std::vector<MAPElite> el = elites();
        shrunk = true;
        ret.reserve(el.size());
        if ( session->runData(archive()->resultIndex).dt == dt ) {
            for ( const MAPElite &e : el )
                ret.push_back(*e.wave);
        } else {
            for ( Stimulation const& I : stimulations() )
                ret.push_back(iStimulation(I, dt));
        }
        break;
    }
    case Subset :
    case Deck:
    case Manual:
        std::vector<Stimulation> stims = stimulations();
        ret.reserve(stims.size());
        for ( Stimulation const& I : stims )
            ret.push_back(iStimulation(I, dt));
        break;
    }

    if ( !shrunk && waveno >= 0 )
        return {ret[waveno]};
    else
        return ret;
}

QString WaveSource::prettyName() const
{
    QString ret;
    switch ( type ) {
    default:        ret = QString("Unknown source type"); break;
    case Archive:   ret = QString("Archive %2 [%3]: %1").arg(archive()->prettyName()); break;
    case Selection: ret = QString("Selection %2 [%3]: %1").arg(selection()->prettyName()); break;
    case Subset:
        if ( archive() ) {
                    ret = QString("Subset %2 [%3]: %1").arg(subset()->prettyName()); break;
        } else {
                    return QString("Subset %2: %1").arg(subset()->prettyName()).arg(idx);
        }
    case Deck:      return QString("Deck %1").arg(idx);
    case Manual:    return QString("Manual %1").arg(idx);
    }
    return ret.arg(idx).arg(archive()->param < 0 ? "EE" : QString::fromStdString(session->project.model().adjustableParams[archive()->param].name));
}

int WaveSource::index() const
{
    int i = 0;
    switch ( type ) {
//    If more source types are added: Iterate up the list, adding the total above at each step like so - don't break:
//    case NewType:
//        i += session->wavesets().manuals().size();
    case Manual:
        i += session->wavesets().decks().size();
    case Deck:
        i += session->wavesets().subsets().size();
    case Subset:
        i += session->wavesets().selections().size();
    case Selection:
        i += session->wavegen().archives().size();
    case Archive:
    default:
        return i + idx;
    }
}

QDataStream &operator<<(QDataStream &os, const WaveSource &src)
{
    os << WaveSource::version << quint32(src.type);
    switch ( src.type ) {
    default:
    case WaveSource::Archive:   os << quint32(src.session->wavegen().archives().size() - src.idx); break;
    case WaveSource::Selection: os << quint32(src.session->wavesets().selections().size() - src.idx); break;
    case WaveSource::Subset:    os << quint32(src.session->wavesets().subsets().size() - src.idx); break;
    case WaveSource::Deck:      os << quint32(src.session->wavesets().decks().size() - src.idx); break;
    case WaveSource::Manual:    os << quint32(src.session->wavesets().manuals().size() - src.idx); break;
    }
    os << qint32(src.waveno);
    return os;
}

QDataStream &operator>>(QDataStream &is, WaveSource &src)
{
    quint32 version, type, idx;
    qint32 waveno;
    is >> version;
    if ( version < 100 ) {
        type = version;
        is >> idx;
        waveno = -1;
    } else {
        is >> type >> idx >> waveno;
    }
    src.type = WaveSource::Type(type);
    switch ( src.type ) {
    default:
    case WaveSource::Archive:   src.idx = src.session->wavegen().archives().size() - idx; break;
    case WaveSource::Selection: src.idx = src.session->wavesets().selections().size() - idx; break;
    case WaveSource::Subset:    src.idx = src.session->wavesets().subsets().size() - idx; break;
    case WaveSource::Deck:      src.idx = src.session->wavesets().decks().size() - idx; break;
    case WaveSource::Manual:    src.idx = src.session->wavesets().manuals().size() - idx; break;
    }
    src.waveno = waveno;
    return is;
}

bool operator==(const WaveSource &lhs, const WaveSource &rhs)
{
    return lhs.type == rhs.type && lhs.idx == rhs.idx && lhs.waveno == rhs.waveno;
}
