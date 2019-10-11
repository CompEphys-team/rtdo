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


#ifndef SAMPLINGPROFILEDIALOG_H
#define SAMPLINGPROFILEDIALOG_H

#include <QWidget>
#include "samplingprofiler.h"
#include "session.h"

namespace Ui {
class SamplingProfileDialog;
}
class QDoubleSpinBox;

class SamplingProfileDialog : public QWidget
{
    Q_OBJECT

public:
    explicit SamplingProfileDialog(Session &s, QWidget *parent = 0);
    ~SamplingProfileDialog();

signals:
    void generate(SamplingProfiler::Profile);

private slots:
    void updateCombo();
    void updatePresets();

    void on_btnStart_clicked();

    void on_btnAbort_clicked();

    void on_btnPreset_clicked();

private:
    Ui::SamplingProfileDialog *ui;
    Session &session;

    std::vector<QDoubleSpinBox*> mins, maxes;

    static constexpr int nHardPresets = 2;

    void setCloseRange(int i);

private:
};

#endif // SAMPLINGPROFILEDIALOG_H
