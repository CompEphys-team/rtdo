#include "wavesource.h"
#include "session.h"

const Wavegen::Archive &WaveSource::archive() const
{
    switch ( type ) {
    default:
    case Archive:   return session->wavegen().archives().at(idx);
    case Selection: return selection()->archive();
    case Subset:    return subset()->src.archive();
    case Deck:      return deck()->sources()[0].archive(); // Return first of multiple archives for lack of a better alternative
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
    }
}

const WaveDeck *WaveSource::deck() const
{
    switch ( type ) {
    default:
    case Archive:
    case Selection:
    case Subset:
        return nullptr;
    case Deck:
        return &session->wavesets().decks().at(idx);
    }
}

std::vector<Stimulation> WaveSource::stimulations() const
{
    std::vector<Stimulation> ret;
    switch ( type ) {
    case Archive :
    {
        ret.reserve(archive().elites.size());
        for ( MAPElite const& e : archive().elites )
            ret.push_back(e.wave);
        break;
    }
    case Selection :
    {
        const WavegenSelection &sel = *selection();
        ret.reserve(sel.size());
        std::vector<size_t> idx(sel.ranges.size());
        for ( size_t i = 0; i < sel.size(); i++ ) {
            for ( int j = sel.ranges.size() - 1; j >= 0; j-- ) {
                if ( ++idx[j] % sel.width(j) == 0 )
                    idx[j] = 0;
                else
                    break;
            }
            bool ok;
            auto it = sel.data_relative(idx, &ok);
            if ( ok )
                ret.push_back(it->wave);
        }
        break;
    }
    case Subset :
        ret = subset()->stimulations();
        break;
    case Deck:
        ret = deck()->stimulations();
        break;
    }
    return ret;
}

QString WaveSource::prettyName() const
{
    QString ret;
    switch ( type ) {
    default:        ret = QString("Unknown source type"); break;
    case Archive:   ret = QString("Archive %2 [%3]: %1").arg(archive().prettyName()); break;
    case Selection: ret = QString("Selection %2 [%3]: %1").arg(selection()->prettyName()); break;
    case Subset:    ret = QString("Subset %2 [%3]: %1").arg(subset()->prettyName()); break;
    case Deck:      return QString("Deck %1").arg(idx);
    }
    return ret.arg(idx).arg(QString::fromStdString(session->project.model().adjustableParams[archive().param].name));
}

int WaveSource::index() const
{
    int i = 0;
    switch ( type ) {
//    If more source types are added: Iterate up the list, adding the total above at each step like so - don't break:
//    case NewType:
//        i += session->wavesetcreator().decks().size();
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
    os << quint32(src.type);
    switch ( src.type ) {
    default:
    case WaveSource::Archive:   os << quint32(src.session->wavegen().archives().size() - src.idx); break;
    case WaveSource::Selection: os << quint32(src.session->wavesets().selections().size() - src.idx); break;
    case WaveSource::Subset:    os << quint32(src.session->wavesets().subsets().size() - src.idx); break;
    case WaveSource::Deck:      os << quint32(src.session->wavesets().decks().size() - src.idx); break;
    }
    return os;
}

QDataStream &operator>>(QDataStream &is, WaveSource &src)
{
    quint32 type, idx;
    is >> type >> idx;
    src.type = WaveSource::Type(type);
    switch ( src.type ) {
    default:
    case WaveSource::Archive:   src.idx = src.session->wavegen().archives().size() - idx; break;
    case WaveSource::Selection: src.idx = src.session->wavesets().selections().size() - idx; break;
    case WaveSource::Subset:    src.idx = src.session->wavesets().subsets().size() - idx; break;
    case WaveSource::Deck:      src.idx = src.session->wavesets().decks().size() - idx; break;
    }
    return is;
}
