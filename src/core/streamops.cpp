#include "streamops.h"

std::istream &operator>>(std::istream &is, QString &str)
{
    std::string tmp;
    is >> tmp;
    str = QString::fromStdString(tmp);
    return is;
}

std::ostream &operator<<(std::ostream &os, const QString &str)
{
    os << str.toStdString();
    return os;
}

QDataStream &operator<<(QDataStream &os, const MAPElite &e)
{
    os << e.wave << e.fitness;
    os << quint32(e.bin.size());
    for ( const size_t &b : e.bin )
        os << quint32(b);
    return os;
}

QDataStream &operator>>(QDataStream &is, MAPElite &e)
{
    is >> e.wave >> e.fitness;
    quint32 bins, val;
    is >> bins;
    e.bin.resize(bins);
    for ( size_t &b : e.bin ) {
        is >> val;
        b = size_t(val);
    }
    return is;
}

QDataStream &operator<<(QDataStream &os, const Stimulation &stim)
{
    os << stim.baseV << stim.duration << stim.tObsBegin << stim.tObsEnd;
    os << quint32(stim.size());
    for ( const Stimulation::Step &s : stim )
        os << s.ramp << s.t << s.V;
    return os;
}

QDataStream &operator>>(QDataStream &is, Stimulation &stim)
{
    // Note: this is floating-point safe under the assumption that is.floatingPointPrecision matches the writer's setting.
    quint32 steps;
    is >> stim.baseV >> stim.duration >> stim.tObsBegin >> stim.tObsEnd;
    is >> steps;
    stim.clear();
    for ( quint32 i = 0; i < steps; i++ ) {
        Stimulation::Step s;
        is >> s.ramp >> s.t >> s.V;
        stim.insert(stim.end(), std::move(s));
    }
    return is;
}

QDataStream &operator<<(QDataStream &os, const iStimulation &stim)
{
    os << stim.baseV << qint32(stim.duration) << qint32(stim.tObsBegin) << qint32(stim.tObsEnd);
    os << quint32(stim.size());
    for ( const iStimulation::Step &s : stim )
        os << s.ramp << qint32(s.t) << s.V;
    return os;
}

QDataStream &operator>>(QDataStream &is, iStimulation &stim)
{
    // Note: this is floating-point safe under the assumption that is.floatingPointPrecision matches the writer's setting.
    quint32 steps;
    qint32 dur, begin, end;
    is >> stim.baseV >> dur >> begin >> end;
    stim.duration = dur;
    stim.tObsBegin = begin;
    stim.tObsEnd = end;
    is >> steps;
    stim.clear();
    for ( quint32 i = 0; i < steps; i++ ) {
        iStimulation::Step s;
        qint32 t;
        is >> s.ramp >> t >> s.V;
        s.t = t;
        stim.insert(stim.end(), std::move(s));
    }
    return is;
}

QDataStream &operator>>(QDataStream &is, MAPElite__scalarStim &e)
{
    is >> e.wave >> e.fitness;
    quint32 bins, val;
    is >> bins;
    e.bin.resize(bins);
    for ( size_t &b : e.bin ) {
        is >> val;
        b = size_t(val);
    }
    return is;
}
