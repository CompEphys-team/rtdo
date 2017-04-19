#include "wavesource.h"
#include "session.h"

const Wavegen::Archive &WaveSource::archive() const
{
    switch ( type ) {
    default:
    case Archive:   return session->wavegen().archives().at(idx);
    case Selection: return session->wavegenselector().selections().at(idx).archive();
    }
}

const WavegenSelection *WaveSource::selection() const
{
    switch ( type ) {
    default:
    case Archive:   return nullptr;
    case Selection: return &session->wavegenselector().selections().at(idx);
    }
}

QString WaveSource::prettyName() const
{
    switch ( type ) {
    default:
    case Archive:   return session->wavegen().prettyName(idx);
    case Selection: return session->wavegenselector().prettyName(idx);
    }
}

int WaveSource::index() const
{
    int i = 0;
    switch ( type ) {
//    If more source types are added: Iterate up the list, adding the total above at each step like so - don't break:
//    case NewType:
//        i += session->wavegenselector().selections.size();
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
    case WaveSource::Selection: os << quint32(src.session->wavegenselector().selections().size() - src.idx); break;
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
    case WaveSource::Selection: src.idx = src.session->wavegenselector().selections().size() - idx; break;
    }
    return is;
}
