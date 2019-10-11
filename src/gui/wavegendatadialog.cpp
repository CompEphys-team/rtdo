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


#include "wavegendatadialog.h"
#include "ui_wavegendatadialog.h"

WavegenDataDialog::WavegenDataDialog(Session &s, int historicIndex, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDataDialog),
    session(s),
    historicIndex(historicIndex)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    if ( historicIndex < 0 ) {
        connect(&session, SIGNAL(wavegenDataChanged()), this, SLOT(importData()));
        connect(this, SIGNAL(apply(WavegenData)), &session, SLOT(setWavegenData(WavegenData)));
    } else {
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close);
    }

    connect(ui->nTrajectories, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &WavegenDataDialog::updateNDetunes);
    connect(ui->trajectoryLength, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &WavegenDataDialog::updateNDetunes);

    importData();
}

WavegenDataDialog::~WavegenDataDialog()
{
    delete ui;
}

void WavegenDataDialog::importData()
{
    WavegenData p = historicIndex < 0 ? session.qWavegenData() : session.wavegenData(historicIndex);
    ui->nInitialWaves->setValue(p.nInitialWaves);
    ui->maxIterations->setValue(p.maxIterations);
    ui->useBaseParameters->setChecked(p.useBaseParameters);

    QStringList str;
    for ( size_t i : p.precisionIncreaseEpochs )
        str << QString::number(i);
    ui->precisionIncreaseEpochs->setText(str.join(','));

    ui->nBinsBubbleTime->setValue(0);
    ui->nBinsBubbleDuration->setValue(0);
    ui->nBinsVoltageDeviation->setValue(0);
    ui->nBinsVoltageIntegral->setValue(0);
    ui->mape_clusterIndex->setChecked(false);
    ui->mape_nClusters->setChecked(false);
    ui->nBinsMeanCurrent->setValue(0);
    for ( const MAPEDimension &dim : p.mapeDimensions ) {
        if ( dim.func == MAPEDimension::Func::BestBubbleTime )
            ui->nBinsBubbleTime->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::BestBubbleDuration)
            ui->nBinsBubbleDuration->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::VoltageDeviation)
            ui->nBinsVoltageDeviation->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::VoltageIntegral)
            ui->nBinsVoltageIntegral->setValue(dim.resolution);
        if ( dim.func == MAPEDimension::Func::EE_ClusterIndex )
            ui->mape_clusterIndex->setChecked(true);
        if ( dim.func == MAPEDimension::Func::EE_NumClusters )
            ui->mape_nClusters->setChecked(true);
        if ( dim.func == MAPEDimension::Func::EE_MeanCurrent ) {
            ui->nBinsMeanCurrent->setValue(dim.resolution);
            ui->meanCurrent_max->setValue(dim.max);
        }
    }
    ui->adjustMaxCurrent->setChecked(p.adjustToMaxCurrent);

    ui->nTrajectories->setValue(p.nTrajectories);
    ui->trajectoryLength->setValue(p.trajectoryLength);
    ui->nDeltabarRuns->setValue(p.nDeltabarRuns);

    ui->cluster_blank->setValue(p.cluster.blank);
    ui->cluster_minLen->setValue(p.cluster.minLen);
    ui->cluster_secLen->setValue(p.cluster.secLen);
    ui->cluster_threshold->setValue(p.cluster.dotp_threshold);
}

void WavegenDataDialog::exportData()
{
    WavegenData p;
    p.nInitialWaves = ui->nInitialWaves->value();
    p.maxIterations = ui->maxIterations->value();
    p.useBaseParameters = ui->useBaseParameters->isChecked();

    std::vector<size_t> prec;
    for ( QString n : ui->precisionIncreaseEpochs->text().split(',', QString::SkipEmptyParts) ) {
        bool ok;
        int num = n.toInt(&ok);
        if ( ok )
            prec.push_back(num);
    }
    std::sort(prec.begin(), prec.end());
    p.precisionIncreaseEpochs = prec;

    std::vector<MAPEDimension> dims;
    const StimulationData &stimd = session.qStimulationData();
    scalar maxDeviation = stimd.maxVoltage-stimd.baseV > stimd.baseV-stimd.minVoltage
            ? stimd.maxVoltage - stimd.baseV
            : stimd.baseV - stimd.minVoltage;
    if ( ui->nBinsBubbleTime->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::BestBubbleTime, 0, stimd.duration, (size_t)ui->nBinsBubbleTime->value()});
    if ( ui->nBinsBubbleDuration->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::BestBubbleDuration, 0, stimd.duration, (size_t)ui->nBinsBubbleDuration->value()});
    if ( ui->nBinsVoltageDeviation->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::VoltageDeviation, 0, maxDeviation, (size_t)ui->nBinsVoltageDeviation->value()});
    if ( ui->nBinsVoltageIntegral->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::VoltageIntegral, 0, maxDeviation * stimd.duration, (size_t)ui->nBinsVoltageIntegral->value()});
    if ( ui->mape_nClusters->isChecked() )
        dims.push_back(MAPEDimension {MAPEDimension::Func::EE_NumClusters, 0, UniversalLibrary::maxClusters, UniversalLibrary::maxClusters});
    if ( ui->mape_clusterIndex->isChecked() )
        dims.push_back(MAPEDimension {MAPEDimension::Func::EE_ClusterIndex, 0, UniversalLibrary::maxClusters, UniversalLibrary::maxClusters});
    if ( ui->nBinsMeanCurrent->value() > 0 )
        dims.push_back(MAPEDimension {MAPEDimension::Func::EE_MeanCurrent, 0, (scalar)ui->meanCurrent_max->value(), (size_t)ui->nBinsMeanCurrent->value()});
    p.mapeDimensions = dims;
    p.adjustToMaxCurrent = ui->adjustMaxCurrent->isChecked();

    p.nTrajectories = ui->nTrajectories->value();
    p.trajectoryLength = ui->trajectoryLength->value();
    p.nDeltabarRuns = ui->nDeltabarRuns->value();

    p.cluster.blank = ui->cluster_blank->value();
    p.cluster.minLen = ui->cluster_minLen->value();
    p.cluster.secLen = ui->cluster_secLen->value();
    p.cluster.dotp_threshold = ui->cluster_threshold->value();

    emit apply(p);
}

void WavegenDataDialog::on_buttonBox_clicked(QAbstractButton *button)
{
    QDialogButtonBox::ButtonRole role = ui->buttonBox->buttonRole(button);
    if ( role  == QDialogButtonBox::AcceptRole ) {
        //ok
        exportData();
        close();
    } else if ( role == QDialogButtonBox::ApplyRole ) {
        // apply
        exportData();
    } else {
        // cancel
        importData();
        close();
    }
}

void WavegenDataDialog::updateNDetunes()
{
    int nTotal = ui->nTrajectories->value() * (ui->trajectoryLength->value() - 1);
    int nParams = session.project.model().nNormalAdjustableParams;
    int nOptions = session.project.model().nOptions;
    int nMinDetunes = nParams * (1 << nOptions) + nOptions - 1;
    ui->nDetunes->setText(QString("%1/%2").arg(nTotal).arg(nMinDetunes));
}
