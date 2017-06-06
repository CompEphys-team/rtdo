#include "gafittersettingsdialog.h"
#include "ui_gafittersettingsdialog.h"

GAFitterSettingsDialog::GAFitterSettingsDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::GAFitterSettingsDialog),
    session(s)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);

    connect(&session, &Session::actionLogged, [=](QString actor, QString action, QString, int) {
        if ( actor == "Config" && action == "cfg" )
            importData();
    });

    connect(ui->targetType, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int idx){
        ui->targetValues->setEnabled(idx==1);
        ui->fits->setEnabled(idx==1);
    });

    auto append = [=](int i){
        const GAFitter::Output &fit = session.gaFitter().results().at(i);
        ui->fits->addItem(QString("Fit %1 (%2 epochs, %3)").arg(i).arg(fit.epochs).arg(fit.deck.prettyName()));
    };
    ui->fits->addItem("Copy values from...");
    for ( size_t i = 0; i < session.gaFitter().results().size(); i++ ) {
        append(i);
    }
    connect(&session.gaFitter(), &GAFitter::done, [=](){
        append(session.gaFitter().results().size()-1);
    });
    connect(ui->fits, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int idx){
        const GAFitter::Output &fit = session.gaFitter().results().at(idx-1);
        for ( int i = 0; i < ui->targetValues->rowCount(); i++ ) {
            qobject_cast<QDoubleSpinBox*>(ui->targetValues->cellWidget(i, 0))->setValue(fit.targets.at(i));
        }
    });

    ui->targetValues->setRowCount(session.project.model().adjustableParams.size());
    QStringList labels;
    for ( int i = 0; i < ui->targetValues->rowCount(); i++ ) {
        labels << QString::fromStdString(session.project.model().adjustableParams.at(i).name);
        QDoubleSpinBox *box = new QDoubleSpinBox();
        box->setDecimals(6);
        box->setRange(session.project.model().adjustableParams.at(i).min, session.project.model().adjustableParams.at(i).max);
        ui->targetValues->setCellWidget(i, 0, box);
    }
    ui->targetValues->setVerticalHeaderLabels(labels);

    importData();

    connect(this, SIGNAL(apply(GAFitterSettings)), &session, SLOT(setGAFitterSettings(GAFitterSettings)));
}

GAFitterSettingsDialog::~GAFitterSettingsDialog()
{
    delete ui;
}

void GAFitterSettingsDialog::importData()
{
    const GAFitterSettings &p = session.gaFitterSettings();
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
    ui->targetType->setCurrentIndex(p.targetType);
    if ( p.targetType == 1 ) {
        for ( int row = 0; row < ui->targetValues->rowCount(); row++ ) {
            qobject_cast<QDoubleSpinBox*>(ui->targetValues->cellWidget(row, 0))->setValue(p.targetValues.at(row));
        }
    }
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
    p.targetType = ui->targetType->currentIndex();
    if ( p.targetType == 1 ) {
        p.targetValues.resize(ui->targetValues->rowCount());
        for ( int row = 0; row < ui->targetValues->rowCount(); row++ ) {
            p.targetValues[row] = qobject_cast<QDoubleSpinBox*>(ui->targetValues->cellWidget(row, 0))->value();
        }
    }
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
