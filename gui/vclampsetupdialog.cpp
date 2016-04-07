/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include "vclampsetupdialog.h"
#include "ui_vclampsetupdialog.h"
#include "util.h"
#include <QFileDialog>
#include "config.h"
#include "run.h"

VClampSetupDialog::VClampSetupDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VClampSetupDialog),
    cinModel(new ChannelListModel(ChannelListModel::AnalogIn)),
    voutModel(new ChannelListModel(ChannelListModel::AnalogOut))
{
    ui->setupUi(this);
    ui->currentInputChannel->setModel(cinModel);
    ui->voltageOutputChannel->setModel(voutModel);
    connect(this, SIGNAL(channelsUpdated()), cinModel, SIGNAL(modelReset()));
    connect(this, SIGNAL(channelsUpdated()), voutModel, SIGNAL(modelReset()));

#ifndef CONFIG_RT
    ui->currentInputChannel->setEnabled(false);
    ui->voltageOutputChannel->setEnabled(false);
#endif
}

VClampSetupDialog::~VClampSetupDialog()
{
    delete ui;
    delete cinModel;
    delete voutModel;
}

void VClampSetupDialog::open()
{
    int i = 0;
    ui->currentInputChannel->setCurrentIndex(-1);
    ui->voltageOutputChannel->setCurrentIndex(-1);
    for ( Channel &c : config->io.channels ) {
        if ( config->vc.in == c.ID() )
            ui->currentInputChannel->setCurrentIndex(i);
        if ( config->vc.out == c.ID() )
            ui->voltageOutputChannel->setCurrentIndex(i);
        ++i;
    }

    ui->gain->setValue(config->vc.gain);
    ui->resistance->setValue(config->vc.resistance);

    ui->waveformFile->setText(QString::fromStdString(config->vc.wavefile));
    ui->popSize->setValue(config->vc.popsize);

    ui->cacheSize->setValue(config->vc.cacheSize);
    ui->cacheUseMedian->setChecked(config->vc.cacheUseMedian);

    QDialog::open();
}

void VClampSetupDialog::accept()
{
    int in = ui->currentInputChannel->currentIndex();
    int out = ui->voltageOutputChannel->currentIndex();
    config->vc.in = ( in >= 0 ) ? config->io.channels.at(in).ID() : 0;
    config->vc.out = ( out >= 0 ) ? config->io.channels.at(out).ID() : 0;
    config->vc.gain = ui->gain->value();
    config->vc.resistance = ui->resistance->value();
    config->vc.wavefile = ui->waveformFile->text().toStdString();
    config->vc.popsize = ui->popSize->value();
    config->vc.cacheSize = (unsigned int)ui->cacheSize->value();
    config->vc.cacheUseMedian = ui->cacheUseMedian->isChecked();

    emit configChanged();

    QDialog::accept();
}

void VClampSetupDialog::on_waveformBrowse_clicked()
{
    QString file, dir;
    dir = dirname(ui->waveformFile->text());
    file = QFileDialog::getOpenFileName(this, QString("Select voltage clamp waveform file..."), dir);
    if ( !file.isEmpty() )
        ui->waveformFile->setText(file);
}
