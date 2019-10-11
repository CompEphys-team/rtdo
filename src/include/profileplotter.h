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


#ifndef PROFILEPLOTTER_H
#define PROFILEPLOTTER_H

#include <QWidget>
#include "session.h"
#include "colorbutton.h"
#include <QCheckBox>

namespace Ui {
class ProfilePlotter;
}

class ProfilePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit ProfilePlotter(Session &session, QWidget *parent = 0);
    ~ProfilePlotter();

private slots:
    void updateProfiles();
    void updateTargets();
    void updateWaves();
    void replot();
    void rescale();

    void clearProfiles();
    void drawProfiles();
    void drawStats();

    void selectSubset();

private:
    Ui::ProfilePlotter *ui;
    Session &session;

    std::vector<QCheckBox*> includes;
    std::vector<ColorButton*> colors;

    bool tickingBoxes;

    void includeWave(size_t waveNo, bool on);
    void paintWave(size_t waveNo, QColor color);

    static constexpr int ValueColumn = 2;
};

#endif // PROFILEPLOTTER_H
