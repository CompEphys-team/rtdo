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


#ifndef WAVEGENFITNESSMAPPER_H
#define WAVEGENFITNESSMAPPER_H

#include <QWidget>
#include "session.h"

namespace Ui {
class WavegenFitnessMapper;
}
class QButtonGroup;
class QDoubleSpinBox;
class QCheckBox;
class QComboBox;
class QSpinBox;
class QCPColorMap;

class WavegenFitnessMapper : public QWidget
{
    Q_OBJECT

public:
    explicit WavegenFitnessMapper(Session &session, QWidget *parent = 0);
    ~WavegenFitnessMapper();

private slots:
    void updateCombo();
    void updateDimensions();

    void replot();

    void on_btnAdd_clicked();

    void on_readMinFitness_clicked();

    void on_pdf_clicked();

    void on_readMaxFitness_clicked();

    void on_deltabar_clicked();

private:
    Ui::WavegenFitnessMapper *ui;
    Session &session;

    QButtonGroup *groupx, *groupy;
    std::vector<QDoubleSpinBox*> mins, maxes;
    std::vector<QCheckBox*> collapse;
    std::vector<QComboBox*> pareto;
    std::vector<QSpinBox*> tolerance;

    std::unique_ptr<WavegenSelection> selection;
    QCPColorMap *colorMap;

    bool select(bool flattenToPlot);

    void initPlot();
};

#endif // WAVEGENFITNESSMAPPER_H
