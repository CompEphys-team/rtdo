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
    QDialog::open();
}

void WavegenSetupDialog::accept()
{
    config->wg.popsize = ui->popsize->value();
    config->wg.ngen = ui->ngen->value();
    QDialog::accept();
}
