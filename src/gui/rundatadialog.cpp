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


#include "rundatadialog.h"
#include "ui_rundatadialog.h"

RunDataDialog::RunDataDialog(Session &s, int historicIndex, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RunDataDialog),
    session(s),
    calibrator(s),
    historicIndex(historicIndex)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    if ( historicIndex < 0 ) {
        connect(&session, &Session::actionLogged, this, [=](QString actor, QString action, QString, int) {
            if ( actor == "Config" && action == "cfg" )
                importData();
        });
        connect(this, SIGNAL(apply(RunData)), &session, SLOT(setRunData(RunData)));
        connect(&session, SIGNAL(runDataChanged()), this, SLOT(importData()));

        connect(ui->measureResistance, &QPushButton::clicked, &calibrator, &Calibrator::findAccessResistance);
        connect(&calibrator, &Calibrator::findingAccessResistance, this, [=](bool done){
            if ( done ) ui->accessResistance->setPrefix("");
            else        ui->accessResistance->setPrefix("... ");
        });
    } else {
        ui->measureResistance->setEnabled(false);
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close);
    }

    importData();
}

RunDataDialog::~RunDataDialog()
{
    delete ui;
}

void RunDataDialog::importData()
{
    RunData p = historicIndex < 0 ? session.qRunData() : session.runData(historicIndex);
    ui->dt->setValue(p.dt);
    ui->VC->setChecked(p.VC);
    ui->simCycles->setValue(p.simCycles);
    ui->clampGain->setValue(p.clampGain);
    ui->accessResistance->setValue(p.accessResistance);
    ui->Imax->setValue(p.Imax);
    ui->settleDuration->setValue(p.settleDuration);
    switch ( p.integrator ) {
    case IntegrationMethod::ForwardEuler:         ui->integrator->setCurrentIndex(0); break;
    case IntegrationMethod::RungeKutta4:          ui->integrator->setCurrentIndex(1); break;
    case IntegrationMethod::RungeKuttaFehlberg45: ui->integrator->setCurrentIndex(2); break;
    }
    ui->noisy->setChecked(p.noisy);
    ui->noisyChannels->setChecked(p.noisyChannels);
    ui->noiseStd->setValue(p.noiseStd);
    ui->noiseTau->setValue(p.noiseTau);
}

void RunDataDialog::exportData()
{
    RunData p;
    p.dt = ui->dt->value();
    p.VC = ui->VC->isChecked();
    p.simCycles = ui->simCycles->value();
    p.clampGain = ui->clampGain->value();
    p.accessResistance = ui->accessResistance->value();
    p.Imax = ui->Imax->value();
    p.settleDuration = ui->settleDuration->value();
    switch ( ui->integrator->currentIndex() ) {
    case 0: p.integrator = IntegrationMethod::ForwardEuler;         break;
    case 1: p.integrator = IntegrationMethod::RungeKutta4;          break;
    case 2: p.integrator = IntegrationMethod::RungeKuttaFehlberg45; break;
    }
    p.noisy = ui->noisy->isChecked();
    p.noisyChannels = ui->noisyChannels->isChecked();
    p.noiseStd = ui->noiseStd->value();
    p.noiseTau = ui->noiseTau->value();
    emit apply(p);
}

void RunDataDialog::on_buttonBox_accepted()
{
    exportData();
    close();
}

void RunDataDialog::on_buttonBox_rejected()
{
    importData();
    close();
}
