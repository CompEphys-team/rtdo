/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-01-20

--------------------------------------------------------------------------*/
#include "include/wavegensetupdialog.h"
#include "ui_wavegensetupdialog.h"
#include "config.h"

WavegenSetupDialog::WavegenSetupDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenSetupDialog)
{
    ui->setupUi(this);
}

WavegenSetupDialog::~WavegenSetupDialog()
{
    delete ui;
}

void WavegenSetupDialog::open()
{
    ui->popsize->setValue(config->wg.popsize);
    ui->ngen->setValue(config->wg.ngen);
    ui->ns_ngenOptimise->setValue(config->wg.ns_ngenOptimise);
    ui->ns_noveltyThreshold->setValue(config->wg.ns_noveltyThreshold);
    ui->ns_optimiseProportion->setValue(config->wg.ns_optimiseProportion);
    ui->tolCurrent->setValue(config->wg.tolCurrent);
    ui->tolDelta->setValue(config->wg.tolDelta);
    ui->tolTime->setValue(config->wg.tolTime);
    ui->tolTime->setSingleStep(config->io.dt);
    QDialog::open();
}

void WavegenSetupDialog::accept()
{
    config->wg.popsize = ui->popsize->value();
    config->wg.ngen = ui->ngen->value();
    config->wg.ns_ngenOptimise = ui->ns_ngenOptimise->value();
    config->wg.ns_noveltyThreshold = ui->ns_noveltyThreshold->value();
    config->wg.ns_optimiseProportion = ui->ns_optimiseProportion->value();
    config->wg.tolCurrent = ui->tolCurrent->value();
    config->wg.tolDelta = ui->tolDelta->value();
    config->wg.tolTime = ui->tolTime->value();
    QDialog::accept();
}
