#include "rundatadialog.h"
#include "ui_rundatadialog.h"

RunDataDialog::RunDataDialog(Session &s, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::RunDataDialog),
    session(s)
{
    ui->setupUi(this);

    connect(&session, &Session::actionLogged, [=](QString actor, QString action, QString, int) {
        if ( actor == "Config" && action == "cfg" )
            importData();
    });
    connect(this, SIGNAL(apply(RunData)), &session, SLOT(setRunData(RunData)));

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
