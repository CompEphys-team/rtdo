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


#ifndef WAVEGENPROGRESSPLOTTER_H
#define WAVEGENPROGRESSPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class WavegenProgressPlotter;
}

class QCheckBox;
struct AbstractGraphProxy
{
    AbstractGraphProxy(QColor color, QCheckBox *cb);
    virtual ~AbstractGraphProxy() {}

    virtual void populate() = 0;
    virtual void extend() = 0;

    QColor color;
    QCheckBox *cb;
    QCPAxis *xAxis = 0, *yAxis = 0;
    const Wavegen::Archive *archive;
    QSharedPointer<QCPGraphDataContainer> dataPtr;
};

class WavegenProgressPlotter : public QWidget
{
    Q_OBJECT

public:
    explicit WavegenProgressPlotter(QWidget *parent = 0);
    void init(Session &session);
    ~WavegenProgressPlotter();

protected slots:
    void updateArchives();
    void searchTick(int);
    void replot();

private:
    Ui::WavegenProgressPlotter *ui;
    Session *session;
    bool inProgress;
    std::vector<AbstractGraphProxy*> proxies;
};

#endif // WAVEGENPROGRESSPLOTTER_H
