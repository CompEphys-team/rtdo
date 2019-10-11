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


#ifndef CALIBRATOR_H
#define CALIBRATOR_H

#include <QObject>
#include "session.h"

class Calibrator : public QObject
{
    Q_OBJECT
public:
    Calibrator(Session &s, QObject *parent = nullptr);

public slots:
    void zeroV1(DAQData p);
    void zeroV2(DAQData p);
    void zeroIin(DAQData p);
    void zeroVout(DAQData p);
    void findAccessResistance();

signals:
    void zeroingV1(bool done);
    void zeroingV2(bool done);
    void zeroingIin(bool done);
    void zeroingVout(bool done);
    void findingAccessResistance(bool done);

protected:
    Session &session;
};

#endif // CALIBRATOR_H
