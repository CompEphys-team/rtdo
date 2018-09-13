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
