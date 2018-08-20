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

    const MetaModel &model = session.project.model();
    ui->constraints->setRowCount(model.adjustableParams.size());
    QStringList labels;
    for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
        labels << QString::fromStdString(model.adjustableParams.at(i).name);

        QDoubleSpinBox *min = new QDoubleSpinBox(), *max = new QDoubleSpinBox(), *fixed = new QDoubleSpinBox();
        min->setDecimals(6);
        max->setDecimals(6);
        fixed->setDecimals(6);
        min->setRange(-1e9, 1e9);
        max->setRange(-1e9, 1e9);
        fixed->setRange(-1e9, 1e9);

        QComboBox *cb = new QComboBox();
        connect(cb, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [=](int idx){
            min->setEnabled(idx==1);
            max->setEnabled(idx==1);
            fixed->setEnabled(idx==2);
            if ( idx == 4 ) { // Reset
                min->setValue(model.adjustableParams[i].min);
                max->setValue(model.adjustableParams[i].max);
                fixed->setValue(model.adjustableParams[i].initial);
                cb->setCurrentIndex(0);
            }
        });
        cb->addItems({"Original", "Range", "Fixed", "Target", "Reset"});

        ui->constraints->setCellWidget(i, 0, cb);
        ui->constraints->setCellWidget(i, 1, fixed);
        ui->constraints->setCellWidget(i, 2, min);
        ui->constraints->setCellWidget(i, 3, max);
    }
    ui->constraints->setVerticalHeaderLabels(labels);
    ui->constraints->setColumnWidth(0, 75);

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
    ui->useLikelihood->setChecked(p.useLikelihood);
    ui->cluster_blank->setValue(p.cluster_blank_after_step);
    ui->cluster_dur->setValue(p.cluster_min_dur);
    ui->cluster_res->setValue(p.cluster_fragment_dur);
    ui->cluster_threshold->setValue(p.cluster_threshold);

    for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
        static_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->setCurrentIndex(p.constraints[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 1))->setValue(p.fixedValue[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->setValue(p.min[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(p.max[i]);
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
    p.useLikelihood = ui->useLikelihood->isChecked();
    p.cluster_blank_after_step = ui->cluster_blank->value();
    p.cluster_min_dur = ui->cluster_dur->value();
    p.cluster_fragment_dur = ui->cluster_res->value();
    p.cluster_threshold = ui->cluster_threshold->value();

    for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
        p.constraints.push_back(static_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->currentIndex());
        p.fixedValue.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 1))->value());
        p.min.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->value());
        p.max.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->value());
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
