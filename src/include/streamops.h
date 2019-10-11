/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


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

QDataStream &operator<<(QDataStream &, const iObservations &);
QDataStream &operator>>(QDataStream &, iObservations&);

// For backcompatibility with pre-iStimulation saves:
struct MAPElite__scalarStim
{
    std::vector<size_t> bin;
    Stimulation wave;
    scalar fitness;
};
QDataStream &operator>>(QDataStream &, MAPElite__scalarStim &);


#endif // STREAMOPS_H
