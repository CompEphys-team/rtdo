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


#ifndef RESPONSEPLOTTER_H
#define RESPONSEPLOTTER_H

#include <QWidget>
#include <QTimer>
#include "queue.h"
#include "types.h"
#include "daq.h"

namespace Ui {
class ResponsePlotter;
}

class ResponsePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ResponsePlotter(QWidget *parent = 0);
    ~ResponsePlotter();

    RTMaybe::Queue<DataPoint> qI, qV, qV2, qO;
    const bool *VC = nullptr;

    //! Pass a DAQ to read I and V from directly, rather than using the DataPoint queues. The output trace is left blank.
    //! Time is deduced from DAQ::samplingDt(), with t=0 for the first sample after a call to clear().
    void setDAQ(DAQ *daq);

public slots:
    void start();
    void stop();
    void clear();

protected slots:
    void replot();

private:
    Ui::ResponsePlotter *ui;
    QTimer dataTimer;
    DAQ *daq = nullptr;
    size_t iT = 0;
};

#endif // RESPONSEPLOTTER_H
