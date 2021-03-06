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


#include "stimulationdatadialog.h"
#include "ui_stimulationdatadialog.h"

StimulationDataDialog::StimulationDataDialog(Session &s, int historicIndex, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::StimulationDataDialog),
    session(s),
    historicIndex(historicIndex)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    if ( historicIndex < 0 ) {
        connect(&session, SIGNAL(stimulationDataChanged()), this, SLOT(importData()));
        connect(this, SIGNAL(apply(StimulationData)), &session, SLOT(setStimulationData(StimulationData)));
        connect(this, SIGNAL(updateWavegenData(WavegenData)), &session, SLOT(setWavegenData(WavegenData)));

        connect(this, SIGNAL(accepted()), this, SLOT(exportData()));
        connect(this, SIGNAL(rejected()), this, SLOT(importData()));
    } else {
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close);
    }

    ui->maxSteps->setMaximum(Stimulation::maxSteps);

    importData();
}

StimulationDataDialog::~StimulationDataDialog()
{
    delete ui;
}

void StimulationDataDialog::importData()
{
    StimulationData p = historicIndex < 0 ? session.qStimulationData() : session.stimulationData(historicIndex);
    ui->baseV->setValue(p.baseV);
    ui->duration->setValue(p.duration);
    ui->minSteps->setValue(p.minSteps);
    ui->maxSteps->setValue(p.maxSteps);
    ui->minV->setValue(p.minVoltage);
    ui->maxV->setValue(p.maxVoltage);
    ui->stepLength->setValue(p.minStepLength);

    ui->mutaN->setValue(p.muta.n);
    ui->mutaStd->setValue(p.muta.std);
    ui->mutaCrossover->setValue(p.muta.lCrossover);
    ui->mutaVoltage->setValue(p.muta.lLevel);
    ui->mutaVoltageStd->setValue(p.muta.sdLevel);
    ui->mutaTime->setValue(p.muta.lTime);
    ui->mutaTimeStd->setValue(p.muta.sdTime);
    ui->mutaNumber->setValue(p.muta.lNumber);
    ui->mutaSwap->setValue(p.muta.lSwap);
    ui->mutaType->setValue(p.muta.lType);

    ui->endWithRamp->setChecked(p.endWithRamp);
}

void StimulationDataDialog::exportData()
{
    StimulationData p;
    p.baseV = ui->baseV->value();
    p.duration = ui->duration->value();
    p.minSteps = ui->minSteps->value();
    p.maxSteps = ui->maxSteps->value();
    p.minVoltage = ui->minV->value();
    p.maxVoltage = ui->maxV->value();
    p.minStepLength = ui->stepLength->value();

    p.muta.n = ui->mutaN->value();
    p.muta.std = ui->mutaStd->value();
    p.muta.lCrossover = ui->mutaCrossover->value();
    p.muta.lLevel = ui->mutaVoltage->value();
    p.muta.sdLevel = ui->mutaVoltageStd->value();
    p.muta.lTime = ui->mutaTime->value();
    p.muta.sdTime = ui->mutaTimeStd->value();
    p.muta.lNumber = ui->mutaNumber->value();
    p.muta.lSwap = ui->mutaSwap->value();
    p.muta.lType = ui->mutaType->value();

    p.endWithRamp = ui->endWithRamp->isChecked();

    WavegenData wd = session.qWavegenData();
    bool wdChanged = false;
    for ( MAPEDimension &m : wd.mapeDimensions ) {
        scalar min = m.min, max = m.max;
        m.setDefaultMinMax(session.qStimulationData(), session.project.model().adjustableParams.size());
        if ( m.min == min && m.max == max ) {
            m.setDefaultMinMax(p, session.project.model().adjustableParams.size());
            wdChanged |= (m.min != min || m.max != max);
        } else { // Keep non-default values unchanged
            m.min = min;
            m.max = max;
        }
    }

    emit apply(p);
    if ( wdChanged )
        emit updateWavegenData(wd);
}
