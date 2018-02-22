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
    ui->simCycles->setValue(p.simCycles);
    ui->clampGain->setValue(p.clampGain);
    ui->accessResistance->setValue(p.accessResistance);
    ui->settleDuration->setValue(p.settleDuration);
    switch ( p.integrator ) {
    case IntegrationMethod::ForwardEuler:         ui->integrator->setCurrentIndex(0); break;
    case IntegrationMethod::RungeKutta4:          ui->integrator->setCurrentIndex(1); break;
    case IntegrationMethod::RungeKuttaFehlberg45: ui->integrator->setCurrentIndex(2); break;
    }
}

void RunDataDialog::exportData()
{
    RunData p;
    p.simCycles = ui->simCycles->value();
    p.clampGain = ui->clampGain->value();
    p.accessResistance = ui->accessResistance->value();
    p.settleDuration = ui->settleDuration->value();
    switch ( ui->integrator->currentIndex() ) {
    case 0: p.integrator = IntegrationMethod::ForwardEuler;         break;
    case 1: p.integrator = IntegrationMethod::RungeKutta4;          break;
    case 2: p.integrator = IntegrationMethod::RungeKuttaFehlberg45; break;
    }
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
