#include "rundatadialog.h"
#include "ui_rundatadialog.h"

RunDataDialog::RunDataDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::RunDataDialog),
    session(s),
    calibrator(s)
{
    ui->setupUi(this);

    connect(&session, &Session::actionLogged, [=](QString actor, QString action, QString, int) {
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

    importData();
}

RunDataDialog::~RunDataDialog()
{
    delete ui;
}

void RunDataDialog::importData()
{
    const RunData &p = session.runData();
    ui->simCycles->setValue(p.simCycles);
    ui->clampGain->setValue(p.clampGain);
    ui->accessResistance->setValue(p.accessResistance);
    ui->settleDuration->setValue(p.settleDuration);
}

void RunDataDialog::exportData()
{
    RunData p;
    p.simCycles = ui->simCycles->value();
    p.clampGain = ui->clampGain->value();
    p.accessResistance = ui->accessResistance->value();
    p.settleDuration = ui->settleDuration->value();
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
