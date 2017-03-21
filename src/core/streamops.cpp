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
    os << double(stim.baseV) << double(stim.duration) << double(stim.tObsBegin) << double(stim.tObsEnd);
    os << quint32(stim.size());
    for ( const Stimulation::Step &s : stim )
        os << s.ramp << double(s.t) << double(s.V);
    return os;
}

QDataStream &operator>>(QDataStream &is, Stimulation &stim)
{
    double baseV, duration, tObsBegin, tObsEnd, t, V;
    quint32 steps;
    is >> baseV >> duration >> tObsBegin >> tObsEnd;
    stim.baseV = baseV;
    stim.duration = duration;
    stim.tObsBegin = tObsBegin;
    stim.tObsEnd = tObsEnd;
    is >> steps;
    stim.clear();
    for ( quint32 i = 0; i < steps; i++ ) {
        Stimulation::Step s;
        is >> s.ramp >> t >> V;
        s.t = t;
        s.V = V;
        stim.insert(stim.end(), std::move(s));
    }
    return is;
}

QDataStream &operator<<(QDataStream &os, const WaveStats &stats)
{
    os << qint32(stats.bubbles) << double(stats.fitness) << qint32(stats.best.cycles) << double(stats.best.tEnd);
    return os;
}

QDataStream &operator>>(QDataStream &is, WaveStats &stats)
{
    double tEnd, fitness;
    qint32 cycles, bubbles;
    is >> bubbles >> fitness >> cycles >> tEnd;
    stats.bubbles = bubbles;
    stats.fitness = fitness;
    stats.best.cycles = cycles;
    stats.best.tEnd = tEnd;
    return is;
}
