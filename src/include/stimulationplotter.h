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


#ifndef STIMULATIONPLOTTER_H
#define STIMULATIONPLOTTER_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"
#include "stimulationgraph.h"

namespace Ui {
class StimulationPlotter;
}

class StimulationPlotter : public QWidget
{
    Q_OBJECT

public:
    explicit StimulationPlotter(QWidget *parent = 0);
    StimulationPlotter(Session &session, QWidget *parent = 0);
    ~StimulationPlotter();

    void init(Session *session);
    void clear();

protected:
    void updateColor(size_t idx, bool replot);

protected slots:
    void resizePanel();
    void updateSources();
    void replot();
    void resizeEvent(QResizeEvent *event);

private slots:
    void on_pdf_clicked();

private:
    Ui::StimulationPlotter *ui;
    Session *session;

    std::vector<StimulationGraph*> graphs;
    std::vector<QColor> colors;

    bool rebuilding;

    WaveSource source;
    Stimulation stim;
};

#endif // STIMULATIONPLOTTER_H
