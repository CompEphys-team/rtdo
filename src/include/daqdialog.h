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


#ifndef DAQDIALOG_H
#define DAQDIALOG_H

#include <QDialog>
#include "session.h"
#include "calibrator.h"

namespace Ui {
class DAQDialog;
}
class QComboBox;
class QDoubleSpinBox;
class QAbstractButton;

class DAQDialog : public QDialog
{
    Q_OBJECT

    struct ChannelUI
    {
        QComboBox *channel;
        QComboBox *range;
        QComboBox *aref;
        QDoubleSpinBox *factor;
        QDoubleSpinBox *offset;
    };

public:
    explicit DAQDialog(Session &s, int historicIndex = -1, QWidget *parent = 0);
    ~DAQDialog();

public slots:
    void importData();
    DAQData exportData();

signals:
    void apply(DAQData);
    void zeroV1(DAQData);
    void zeroV2(DAQData);
    void zeroIin(DAQData);
    void zeroVout(DAQData);

protected slots:
    DAQData getFormData();
    void updateChannelCapabilities(int tab = -1, bool checkDevice = true);
    void updateSingleChannelCapabilities(void *vdev, int subdev, ChannelUI &cui);

private slots:
    void on_buttonBox_clicked(QAbstractButton *button);

private:
    Ui::DAQDialog *ui;
    Session &session;
    Calibrator calibrator;
    int historicIndex;

    ChannelUI chanUI[5];
};

#endif // DAQDIALOG_H
