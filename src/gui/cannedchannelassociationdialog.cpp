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

    ui->cbCurrent->setCurrentIndex(CannedDAQ::Iidx + 1);
    ui->cbVoltage->setCurrentIndex(CannedDAQ::Vidx + 1);
    ui->cbVoltage2->setCurrentIndex(CannedDAQ::V2idx + 1);
    ui->scaleCurrent->setValue(CannedDAQ::Iscale);
    ui->scaleVoltage->setValue(CannedDAQ::Vscale);
    ui->scaleVoltage2->setValue(CannedDAQ::V2scale);
}

CannedChannelAssociationDialog::~CannedChannelAssociationDialog()
{
    delete ui;
}

void CannedChannelAssociationDialog::on_CannedChannelAssociationDialog_accepted()
{
    CannedDAQ::Iidx = ui->cbCurrent->currentIndex() - 1;
    CannedDAQ::Vidx = ui->cbVoltage->currentIndex() - 1;
    CannedDAQ::V2idx = ui->cbVoltage2->currentIndex() - 1;

    CannedDAQ::Iscale = ui->scaleCurrent->value();
    CannedDAQ::Vscale = ui->scaleVoltage->value();
    CannedDAQ::V2scale = ui->scaleVoltage2->value();
}
