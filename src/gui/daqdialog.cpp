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


#include "daqdialog.h"
#include "ui_daqdialog.h"
#include <comedilib.h>
#include <QMessageBox>
#include <QStandardItemModel>

static std::vector<FilterMethod> methods;

DAQDialog::DAQDialog(Session &s, int historicIndex, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::DAQDialog),
    session(s),
    calibrator(s),
    historicIndex(historicIndex)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    chanUI[0] = {ui->channel, ui->range, ui->reference, ui->conversionFactor, ui->offset}; // Voltage in
    chanUI[1] = {ui->channel_2, ui->range_2, ui->reference_2, ui->conversionFactor_2, ui->offset_2}; // Current in
    chanUI[2] = {ui->channel_5, ui->range_5, ui->reference_5, ui->conversionFactor_5, ui->offset_5}; // V2
    chanUI[3] = {ui->channel_3, ui->range_3, ui->reference_3, ui->conversionFactor_3, ui->offset_3}; // Voltage cmd
    chanUI[4] = {ui->channel_4, ui->range_4, ui->reference_4, ui->conversionFactor_4, ui->offset_4}; // Current cmd

    if ( historicIndex >= 0 ) {
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close);
        ui->calibrate->setEnabled(false);
        ui->calibrate_2->setEnabled(false);
        ui->calibrate_3->setEnabled(false);
        ui->calibrate_5->setEnabled(false);
    } else {
        ui->buttonBox->addButton("Apply and set as project default", QDialogButtonBox::ApplyRole);

        connect(&session, SIGNAL(DAQDataChanged()), this, SLOT(importData()));
        connect(this, SIGNAL(apply(DAQData)), &session, SLOT(setDAQData(DAQData)));

        connect(ui->deviceNumber, SIGNAL(valueChanged(QString)), this, SLOT(updateChannelCapabilities()));
        for ( int i = 0; i < 5; i++ ) {
            connect(chanUI[i].channel, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [=](int){
                updateChannelCapabilities(i, false);
            });
        }
    }

    connect(ui->calibrate, &QPushButton::clicked, this, [=](bool){ emit zeroV1(getFormData()); });
    connect(this, SIGNAL(zeroV1(DAQData)), &calibrator, SLOT(zeroV1(DAQData)));
    connect(&calibrator, &Calibrator::zeroingV1, this, [=](bool done){
        if ( done ) ui->offset->setPrefix("");
        else        ui->offset->setPrefix("... ");
    });
    connect(ui->calibrate_2, &QPushButton::clicked, this, [=](bool){ emit zeroIin(getFormData()); });
    connect(this, SIGNAL(zeroIin(DAQData)), &calibrator, SLOT(zeroIin(DAQData)));
    connect(&calibrator, &Calibrator::zeroingIin, this, [=](bool done){
        if ( done ) ui->offset_2->setPrefix("");
        else        ui->offset_2->setPrefix("... ");
    });
    connect(ui->calibrate_3, &QPushButton::clicked, this, [=](bool){ emit zeroVout(getFormData()); });
    connect(this, SIGNAL(zeroVout(DAQData)), &calibrator, SLOT(zeroVout(DAQData)));
    connect(&calibrator, &Calibrator::zeroingVout, this, [=](bool done){
        if ( done ) ui->offset_3->setPrefix("");
        else        ui->offset_3->setPrefix("... ");
    });
    connect(ui->calibrate_5, &QPushButton::clicked, this, [=](bool){ emit zeroV2(getFormData()); });
    connect(this, SIGNAL(zeroV2(DAQData)), &calibrator, SLOT(zeroV2(DAQData)));
    connect(&calibrator, &Calibrator::zeroingV2, this, [=](bool done){
        if ( done ) ui->offset_5->setPrefix("");
        else        ui->offset_5->setPrefix("... ");
    });

    methods = {FilterMethod::MovingAverage, FilterMethod::SavitzkyGolay23, FilterMethod::SavitzkyGolay45};
    QStringList labels;
    for ( FilterMethod m : methods )
        labels << QString::fromStdString(toString(m));
    ui->filterMethod->addItems(labels);

    ui->timeout->setSpecialValueText("No timeout");

    connect(ui->source, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [=](int idx){
        if ( idx < 0 )  return;
        ui->sourceStack->setCurrentIndex(std::min(idx, 2));
        if ( idx > 1 ) {
            const MetaModel &model = session.project.model(idx-2);
            ui->targetValues->clear();
            ui->targetValues->setRowCount(model.adjustableParams.size());
            QStringList labels;
            for ( int i = 0; i < ui->targetValues->rowCount(); i++ ) {
                labels << QString::fromStdString(model.adjustableParams.at(i).name);
                QDoubleSpinBox *box = new QDoubleSpinBox();
                box->setDecimals(6);
                box->setRange(model.adjustableParams.at(i).min, model.adjustableParams.at(i).max);
                box->setValue(model.adjustableParams.at(i).initial);
                ui->targetValues->setCellWidget(i, 0, box);
            }
            ui->targetValues->setVerticalHeaderLabels(labels);
        }
    });
    connect(ui->targetType, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [=](int idx){
        ui->targetValues->setEnabled(idx == 2); // Enable manual value entry on fixed values only
    });

    for ( const MetaModel &model : session.project.extraModels() )
        ui->source->addItem(QString::fromStdString(model.name()));

    importData();
}

DAQDialog::~DAQDialog()
{
    delete ui;
}

void DAQDialog::importData()
{
    DAQData p = historicIndex < 0 ? session.qDaqData() : session.daqData(historicIndex);

    ui->source->setCurrentIndex(-1); // Force dependent sourceStack update
    ui->source->setCurrentIndex(p.simulate + 1);
    ui->deviceNumber->setValue(p.devNo);
    ui->throttle->setValue(p.throttle);

    updateChannelCapabilities(-1, false);

    const ChnData *cp[] = {&p.voltageChn, &p.currentChn, &p.V2Chan, &p.vclampChan, &p.cclampChan};
    for ( int i = 0; i < 5; i++ ) {
        chanUI[i].channel->setCurrentIndex(cp[i]->idx);
        chanUI[i].range->setCurrentIndex(cp[i]->range);
        chanUI[i].aref->setCurrentIndex(cp[i]->aref);
        chanUI[i].factor->setValue(cp[i]->gain);
        chanUI[i].offset->setValue(cp[i]->offset);
    }

    ui->outputResolution->setValue(p.simd.outputResolution);
    ui->noise->setChecked(p.simd.noise);
    ui->noiseStd->setValue(p.simd.noiseStd);
    ui->noiseTau->setValue(p.simd.noiseTau);

    ui->targetType->setCurrentIndex(p.simd.paramSet);
    if ( p.simulate > 0 ) {
        if ( p.simd.paramSet == 2 ) {
            for ( size_t i = 0; i < p.simd.paramValues.size(); i++ ) {
                QDoubleSpinBox *cell = qobject_cast<QDoubleSpinBox*>(ui->targetValues->cellWidget(i, 0));
                if ( p.simd.paramValues[i] < cell->minimum() )
                    cell->setMinimum(p.simd.paramValues[i] - 1);
                if ( p.simd.paramValues[i] < cell->maximum() )
                    cell->setMaximum(p.simd.paramValues[i] + 1);
                cell->setValue(p.simd.paramValues[i]);
            }
        } else {
            const MetaModel &model = session.project.model(p.simulate-1);
            for ( size_t i = 0; i < model.adjustableParams.size(); i++ ) {
                qobject_cast<QDoubleSpinBox*>(ui->targetValues->cellWidget(i, 0))->setValue(model.adjustableParams[i].initial);
            }
        }
    }

    ui->cache->setChecked(p.cache.active);
    ui->numTraces->setValue(p.cache.numTraces);
    ui->useMedian->setChecked(p.cache.useMedian);
    ui->timeout->setValue(p.cache.timeout * 1e-3); // ms -> s

    ui->filter->setChecked(p.filter.active);
    ui->samplesPerDt->setValue(p.filter.samplesPerDt);
    ui->filterWidth->setValue(p.filter.width);
    ui->filterMethod->setCurrentText(QString::fromStdString(toString(p.filter.method)));
}

DAQData DAQDialog::getFormData()
{
    DAQData p;

    p.simulate = ui->source->currentIndex() - 1;
    p.devNo = ui->deviceNumber->value();
    p.throttle = ui->throttle->value();

    ChnData *cp[] = {&p.voltageChn, &p.currentChn, &p.V2Chan, &p.vclampChan, &p.cclampChan};
    for ( int i = 0; i < 5; i++ ) {
        int idx = chanUI[i].channel->currentIndex();
        cp[i]->idx = idx < 0 ? 0 : idx;
        cp[i]->active = idx >= 0;
        cp[i]->range = chanUI[i].range->currentIndex();
        cp[i]->aref = chanUI[i].aref->currentIndex();
        cp[i]->gain = chanUI[i].factor->value();
        cp[i]->offset = chanUI[i].offset->value();
    }

    p.simd.outputResolution = ui->outputResolution->value();
    p.simd.noise = ui->noise->isChecked();
    p.simd.noiseStd = ui->noiseStd->value();
    p.simd.noiseTau = ui->noiseTau->value();
    p.simd.paramSet = ui->targetType->currentIndex();
    if ( p.simd.paramSet == 2 ) {
        p.simd.paramValues.resize(ui->targetValues->rowCount());
        for ( size_t i = 0; i < p.simd.paramValues.size(); i++ )
            p.simd.paramValues[i] = qobject_cast<QDoubleSpinBox*>(ui->targetValues->cellWidget(i, 0))->value();
    } else {
        p.simd.paramValues = (historicIndex < 0 ? session.qDaqData() : session.daqData(historicIndex)).simd.paramValues;
    }

    p.cache.active = ui->cache->isChecked();
    p.cache.numTraces = ui->numTraces->value();
    p.cache.useMedian = ui->useMedian->isChecked();
    p.cache.timeout = ui->timeout->value() * 1e3; // s -> ms

    p.filter.active = ui->filter->isChecked();
    p.filter.samplesPerDt = ui->samplesPerDt->value();
    p.filter.width = ui->filterWidth->value();
    if ( p.filter.width % 2 == 0 )
        ui->filterWidth->setValue(++p.filter.width);
    for ( FilterMethod m : methods ) {
        if ( QString::fromStdString(toString(m)) == ui->filterMethod->currentText() ) {
            p.filter.method = m;
            break;
        }
    }

    return p;
}

DAQData DAQDialog::exportData()
{
    DAQData p = getFormData();
    emit apply(p);
    return p;
}

void DAQDialog::updateChannelCapabilities(int tab, bool checkDevice)
{
    DAQData p;
    p.devNo = ui->deviceNumber->value();
    comedi_t *dev = comedi_open(p.devname().c_str());
    if ( dev ) {
        ui->deviceName->setText(comedi_get_board_name(dev));
        int aidev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AI, 0);
        int aodev = comedi_find_subdevice_by_type(dev, COMEDI_SUBD_AO, 0);
        int subdev[5] = {aidev, aidev, aidev, aodev, aodev};
        if ( tab < 0 )
            for ( int i = 0; i < 5; i++ )
                updateSingleChannelCapabilities(dev, subdev[i], chanUI[i]);
        else
            updateSingleChannelCapabilities(dev, subdev[tab], chanUI[tab]);
    } else if ( checkDevice && isVisible() && ui->targetType->currentIndex() == 0 ) {
        ui->deviceName->setText("No such device");
        QMessageBox err;
        err.setText(QString("Unable to find or open device \"%1\"\n%2")
                    .arg(QString::fromStdString(p.devname()))
                    .arg(comedi_strerror(comedi_errno())));
        err.exec();
    }
}

void DAQDialog::updateSingleChannelCapabilities(void *vdev, int subdev, DAQDialog::ChannelUI &cui)
{
    comedi_t *dev = (comedi_t*)vdev;
    int currentChannel = cui.channel->currentIndex();

    int nChannels = comedi_get_n_channels(dev, subdev);
    if ( nChannels != cui.channel->count() ) {
        QStringList labels;
        for ( int i = 0; i < nChannels; i++ ) {
            labels << QString::number(i);
        }
        cui.channel->clear();
        cui.channel->addItems(labels);
        currentChannel = currentChannel < 0 || currentChannel > labels.size() ? 0 : currentChannel;
        cui.channel->setCurrentIndex(currentChannel);
    }

    int nRanges = comedi_get_n_ranges(dev, subdev, currentChannel);
    {
        QStringList labels;
        for ( int i = 0; i < nRanges; i++ ) {
            comedi_range *range = comedi_get_range(dev, subdev, currentChannel, i);
            QString unit;
            if ( range->unit == UNIT_mA ) unit = "mA";
            else if ( range->unit == UNIT_volt ) unit = "V";
            labels << QString("%1 to %2 %3").arg(range->min).arg(range->max).arg(unit);
        }
        int currentIdx = cui.range->currentIndex();
        cui.range->clear();
        cui.range->addItems(labels);
        cui.range->setCurrentIndex(currentIdx < 0 || currentIdx > labels.size() ? 0 : currentIdx);
    }

    unsigned int sdf = comedi_get_subdevice_flags(dev, subdev);
    {
        unsigned int ref[4] = {SDF_GROUND, SDF_COMMON, SDF_DIFF, SDF_OTHER};
        QStandardItemModel *model = qobject_cast<QStandardItemModel*>(cui.aref->model());
        for ( int i = 0; i < 4; i++ ) {
            QStandardItem *item = model->item(i);
            item->setFlags((sdf & ref[i]) ? (item->flags() | Qt::ItemIsEnabled) : (item->flags() & ~Qt::ItemIsEnabled));
        }
    }
}

void DAQDialog::on_buttonBox_clicked(QAbstractButton *button)
{
    QDialogButtonBox::ButtonRole role = ui->buttonBox->buttonRole(button);
    if ( role == QDialogButtonBox::AcceptRole ) { // OK
        exportData();
        close();
    } else if ( role == QDialogButtonBox::ApplyRole ) { // Apply & set project default
        session.project.setDaqData(exportData());
        close();
    } else { // Cancel
        importData();
        close();
    }
}
