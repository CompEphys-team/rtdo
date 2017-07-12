#include "stimulationdatadialog.h"
#include "ui_stimulationdatadialog.h"

StimulationDataDialog::StimulationDataDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::StimulationDataDialog),
    session(s)
{
    ui->setupUi(this);
    connect(&session, SIGNAL(stimulationDataChanged()), this, SLOT(importData()));
    connect(this, SIGNAL(apply(StimulationData)), &session, SLOT(setStimulationData(StimulationData)));

    connect(this, SIGNAL(accepted()), this, SLOT(exportData()));
    connect(this, SIGNAL(rejected()), this, SLOT(importData()));

    importData();
}

StimulationDataDialog::~StimulationDataDialog()
{
    delete ui;
}

void StimulationDataDialog::importData()
{
    const StimulationData &p = session.stimulationData();
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
    emit apply(p);
}
