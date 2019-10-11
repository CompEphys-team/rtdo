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


#ifndef SAMPLINGPROFILEPLOTTER_H
#define SAMPLINGPROFILEPLOTTER_H

#include <QWidget>
#include <QButtonGroup>
#include <QCheckBox>
#include "session.h"

namespace Ui {
class SamplingProfilePlotter;
}

class SamplingProfilePlotter : public QWidget
{
    Q_OBJECT

public:
    explicit SamplingProfilePlotter(Session &s, QWidget *parent = 0);
    ~SamplingProfilePlotter();

protected:
    double value(int i, int dimension,
                 const SamplingProfiler::Profile &prof,
                 const std::vector<MAPElite> &elites,
                 const std::vector<MAPEDimension> &dim);

    enum class Selection {None, Plot, Data};

protected slots:
    void updateTable();
    void updateProfiles();
    void setProfile(int);
    void replot(Selection sel = Selection::Plot, bool showAll = false);
    void hideUnselected();
    void showAll();

private slots:
    void on_pdf_clicked();

    void on_pareto_clicked();

    void on_addDeckGo_clicked();

private:
    Ui::SamplingProfilePlotter *ui;
    Session &session;

    static constexpr int nFixedColumns = 9;

    bool updating;

    struct DataPoint {
        double key, value;
        size_t idx;
        bool selected;
        bool hidden;
    };
    std::vector<DataPoint> points;
    std::vector<double> minima, maxima;

    std::vector<QButtonGroup *> paretoGroups;
    std::vector<QCheckBox *> scoreChecks;
};

#endif // SAMPLINGPROFILEPLOTTER_H
