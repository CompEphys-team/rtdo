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


#ifndef DEVIATIONBOXPLOT_H
#define DEVIATIONBOXPLOT_H

#include <QWidget>
#include "fitinspector.h"

namespace Ui {
class DeviationBoxPlot;
}

class DeviationBoxPlot : public QWidget
{
    Q_OBJECT

public:
    explicit DeviationBoxPlot(QWidget *parent = 0);
    ~DeviationBoxPlot();

    void init(Session *session);

    void setData(std::vector<FitInspector::Group> data, bool summarising);

public slots:
    void replot();

private slots:
    void on_boxplot_pdf_clicked();

private:
    Ui::DeviationBoxPlot *ui;
    Session *session;

    bool summarising = false;
    std::vector<FitInspector::Group> data;
};

#endif // DEVIATIONBOXPLOT_H
