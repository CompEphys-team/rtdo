#ifndef STREAMOPS_H
#define STREAMOPS_H

#include <QString>
#include <QDataStream>
#include <iostream>
#include "types.h"

std::istream &operator>>(std::istream &is, QString &str);
std::ostream &operator<<(std::ostream &os, const QString &str);

QDataStream &operator<<(QDataStream &, const MAPElite &);
QDataStream &operator>>(QDataStream &, MAPElite &);

QDataStream &operator<<(QDataStream &, const Stimulation &);
QDataStream &operator>>(QDataStream &, Stimulation &);

QDataStream &operator<<(QDataStream &, const iStimulation &);
QDataStream &operator>>(QDataStream &, iStimulation &);

// For backcompatibility with pre-iStimulation saves:
struct MAPElite__scalarStim
{
    std::vector<size_t> bin;
    Stimulation wave;
    scalar fitness;
};
QDataStream &operator>>(QDataStream &, MAPElite__scalarStim &);


#endif // STREAMOPS_H
