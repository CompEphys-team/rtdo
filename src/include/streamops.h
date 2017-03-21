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

QDataStream &operator<<(QDataStream &, const WaveStats &);
QDataStream &operator>>(QDataStream &, WaveStats &);


#endif // STREAMOPS_H
