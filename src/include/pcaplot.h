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


#ifndef PCAPLOT_H
#define PCAPLOT_H

#include <QWidget>
#include "qcustomplot.h"
#include "universallibrary.h"

namespace Ui {
class PCAPlot;
}

class PCAPlot : public QWidget
{
    Q_OBJECT

public:
    explicit PCAPlot(QWidget *parent = 0);
    PCAPlot(Session &s, QWidget *parent = 0);
    ~PCAPlot();

    void init(const UniversalLibrary *lib);

public slots:
    void replot();

private slots:
    void compute();

    void on_pdf_clicked();

private:
    Ui::PCAPlot *ui;
    Session *session;
    const UniversalLibrary *lib;
    QSharedPointer<QCPGraphDataContainer> data;
    size_t x = 0, y = 1;
};

#endif // PCAPLOT_H
