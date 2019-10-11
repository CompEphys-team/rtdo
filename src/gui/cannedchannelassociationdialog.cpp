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


#include "cannedchannelassociationdialog.h"
#include "ui_cannedchannelassociationdialog.h"

CannedChannelAssociationDialog::CannedChannelAssociationDialog(Session &s, CannedDAQ *daq, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::CannedChannelAssociationDialog),
    session(s),
    daq(daq)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    for ( const QuotedString &name : daq->channelNames ) {
        ui->cbCurrent->addItem(QString::fromStdString(name));
        ui->cbVoltage->addItem(QString::fromStdString(name));
        ui->cbVoltage2->addItem(QString::fromStdString(name));
    }

    ui->cbCurrent->setCurrentIndex(session.cdaq_assoc.Iidx + 1);
    ui->cbVoltage->setCurrentIndex(session.cdaq_assoc.Vidx + 1);
    ui->cbVoltage2->setCurrentIndex(session.cdaq_assoc.V2idx + 1);
    ui->scaleCurrent->setValue(session.cdaq_assoc.Iscale);
    ui->scaleVoltage->setValue(session.cdaq_assoc.Vscale);
    ui->scaleVoltage2->setValue(session.cdaq_assoc.V2scale);
}

CannedChannelAssociationDialog::~CannedChannelAssociationDialog()
{
    delete ui;
}

void CannedChannelAssociationDialog::on_CannedChannelAssociationDialog_accepted()
{
    session.cdaq_assoc.Iidx = ui->cbCurrent->currentIndex() - 1;
    session.cdaq_assoc.Vidx = ui->cbVoltage->currentIndex() - 1;
    session.cdaq_assoc.V2idx = ui->cbVoltage2->currentIndex() - 1;

    session.cdaq_assoc.Iscale = ui->scaleCurrent->value();
    session.cdaq_assoc.Vscale = ui->scaleVoltage->value();
    session.cdaq_assoc.V2scale = ui->scaleVoltage2->value();
}
