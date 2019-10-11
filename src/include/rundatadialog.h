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


#ifndef RUNDATADIALOG_H
#define RUNDATADIALOG_H

#include <QDialog>
#include "session.h"
#include "calibrator.h"

namespace Ui {
class RunDataDialog;
}
class QAbstractButton;

class RunDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit RunDataDialog(Session &s, int historicIndex = -1, QWidget *parent = 0);
    ~RunDataDialog();

public slots:
    void importData();
    void exportData();

private slots:
    void on_buttonBox_accepted();
    void on_buttonBox_rejected();

private:
    Ui::RunDataDialog *ui;
    Session &session;
    Calibrator calibrator;
    int historicIndex;

signals:
    void apply(RunData);
};

#endif // RUNDATADIALOG_H
