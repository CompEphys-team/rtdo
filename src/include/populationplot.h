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


#ifndef POPULATIONPLOT_H
#define POPULATIONPLOT_H

#include <QWidget>
#include "session.h"
#include "qcustomplot.h"

namespace Ui {
class PopulationPlot;
}

class PopulationPlot : public QWidget
{
    Q_OBJECT

public:
    explicit PopulationPlot(QWidget *parent = 0);
    PopulationPlot(Session &session, QWidget *parent = 0);
    ~PopulationPlot();

    void init(Session *session, bool enslave);
    void clear();

public slots:
    void replot();
    void updateCombos();

protected slots:
    void resizeEvent(QResizeEvent *event);
    void resizePanel();
    void clearPlotLayout();
    void buildPlotLayout();
    void xRangeChanged(QCPRange);

private slots:
    void on_pdf_clicked();

private:
    Ui::PopulationPlot *ui;
    Session *session;
    UniversalLibrary *lib = nullptr;

    std::vector<QCPAxisRect*> axRects;
    QCPColorScale *scaleBar;

    bool enslaved = false;
};

#endif // POPULATIONPLOT_H
