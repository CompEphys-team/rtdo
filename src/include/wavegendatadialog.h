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


#ifndef WAVEGENDATADIALOG_H
#define WAVEGENDATADIALOG_H

#include <QDialog>
#include "session.h"

namespace Ui {
class WavegenDataDialog;
}
class QAbstractButton;

class WavegenDataDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WavegenDataDialog(Session &s, int historicIndex = -1, QWidget *parent = 0);
    ~WavegenDataDialog();

public slots:
    void importData();
    void exportData();

signals:
    void apply(WavegenData);

private slots:
    void on_buttonBox_clicked(QAbstractButton *button);
    void updateNDetunes();

private:
    Ui::WavegenDataDialog *ui;
    Session &session;
    int historicIndex;
};

#endif // WAVEGENDATADIALOG_H
