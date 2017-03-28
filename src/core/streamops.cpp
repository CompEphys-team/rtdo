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
    os << e.wave << e.stats;
    os << quint32(e.bin.size());
    for ( const size_t &b : e.bin )
        os << quint32(b);
    return os;
}

QDataStream &operator>>(QDataStream &is, MAPElite &e)
{
    is >> e.wave >> e.stats;
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

QDataStream &operator<<(QDataStream &os, const WaveStats &stats)
{
    os << qint32(stats.bubbles) << stats.fitness << qint32(stats.best.cycles) << stats.best.tEnd;
    return os;
}

QDataStream &operator>>(QDataStream &is, WaveStats &stats)
{
    qint32 cycles, bubbles;
    is >> bubbles >> stats.fitness >> cycles >> stats.best.tEnd;
    stats.bubbles = bubbles;
    stats.best.cycles = cycles;
    return is;
}