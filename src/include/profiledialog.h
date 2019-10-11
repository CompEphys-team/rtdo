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


#ifndef PROFILEDIALOG_H
#define PROFILEDIALOG_H

#include <QDialog>
#include "errorprofiler.h"
#include "wavegendialog.h"
#include "session.h"

namespace Ui {
class ProfileDialog;
}
class QSpinBox;
class QDoubleSpinBox;

class ProfileDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ProfileDialog(Session &s, QWidget *parent = 0);
    ~ProfileDialog();

signals:
    void generate();

private slots:
    void updateCombo();
    void updatePresets();

    void on_btnStart_clicked();

    void on_btnAbort_clicked();

    void on_btnPreset_clicked();

private:
    Ui::ProfileDialog *ui;
    Session &session;

    std::vector<QSpinBox*> ns;
    std::vector<QDoubleSpinBox*> mins, maxes;

    static constexpr int nHardPresets = 6;

    void setCloseRange(int i);
};

#endif // PROFILEDIALOG_H
