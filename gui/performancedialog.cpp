/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-04

--------------------------------------------------------------------------*/
#include "performancedialog.h"
#include "ui_performancedialog.h"
#include "config.h"
#include "realtimeenvironment.h"

PerformanceDialog::PerformanceDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PerformanceDialog)
{
    ui->setupUi(this);
}

PerformanceDialog::~PerformanceDialog()
{
    delete ui;
}

void PerformanceDialog::open()
{
    ui->ai_supersampling->setValue(config->io.ai_supersampling);
    ui->simCycles->setValue(config->model.cycles);
    QDialog::open();
}

void PerformanceDialog::accept()
{
    apply();
    QDialog::accept();
}

void PerformanceDialog::on_buttonBox_clicked(QAbstractButton *button)
{
    if ( ui->buttonBox->buttonRole(button) == QDialogButtonBox::ApplyRole ) {
        apply();
    }
}

void PerformanceDialog::apply()
{
    int sup = ui->ai_supersampling->value();
    config->io.ai_supersampling = sup;
    RealtimeEnvironment::env().setSupersamplingRate(sup);

    config->model.cycles = ui->simCycles->value();
}

void PerformanceDialog::on_reportingToggle_clicked()
{
    RealtimeEnvironment::env().setIdleTimeReporting(!(RealtimeEnvironment::env().idleTimeReporting()));
}
