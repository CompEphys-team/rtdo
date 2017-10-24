#include "gafittersettingsdialog.h"
#include "ui_gafittersettingsdialog.h"

GAFitterSettingsDialog::GAFitterSettingsDialog(Session &s, int historicIndex, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GAFitterSettingsDialog),
    session(s),
    historicIndex(historicIndex)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);

    if ( historicIndex < 0 ) {
        connect(&session, &Session::actionLogged, this, [=](QString actor, QString action, QString, int) {
            if ( actor == "Config" && action == "cfg" )
                importData();
        });
        connect(this, SIGNAL(apply(GAFitterSettings)), &session, SLOT(setGAFitterSettings(GAFitterSettings)));
    } else {
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close);
    }

    importData();
}

GAFitterSettingsDialog::~GAFitterSettingsDialog()
{
    delete ui;
}

void GAFitterSettingsDialog::importData()
{
    GAFitterSettings p = historicIndex < 0 ? session.qGaFitterSettings() : session.gaFitterSettings(historicIndex);
    ui->maxEpochs->setValue(p.maxEpochs);
    ui->randomOrder->setCurrentIndex(p.randomOrder);
    ui->orderBiasDecay->setValue(p.orderBiasDecay);
    ui->orderBiasStartEpoch->setValue(p.orderBiasStartEpoch);
    ui->nElites->setValue(p.nElite);
    ui->nReinit->setValue(p.nReinit);
    ui->crossover->setValue(p.crossover);
    ui->decaySigma->setChecked(p.decaySigma);
    ui->sigmaHalflife->setValue(p.sigmaHalflife);
    ui->sigmaInitial->setValue(p.sigmaInitial);
}

void GAFitterSettingsDialog::exportData()
{
    GAFitterSettings p;
    p.maxEpochs = ui->maxEpochs->value();
    p.randomOrder = ui->randomOrder->currentIndex();
    p.orderBiasDecay = ui->orderBiasDecay->value();
    p.orderBiasStartEpoch = ui->orderBiasStartEpoch->value();
    p.nElite = ui->nElites->value();
    p.nReinit = ui->nReinit->value();
    p.crossover = ui->crossover->value();
    p.decaySigma = ui->decaySigma->isChecked();
    p.sigmaHalflife = ui->sigmaHalflife->value();
    p.sigmaInitial = ui->sigmaInitial->value();
    emit apply(p);
}

void GAFitterSettingsDialog::on_buttonBox_clicked(QAbstractButton *button)
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
