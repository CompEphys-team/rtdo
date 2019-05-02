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
        const AdjustableParam &p = model.adjustableParams.at(i);
        labels << QString("%1 %2").arg(QString(p.multiplicative ? '*' : '+'), QString::fromStdString(p.name));

        QDoubleSpinBox *min = new QDoubleSpinBox(), *max = new QDoubleSpinBox(), *fixed = new QDoubleSpinBox(), *sigma = new QDoubleSpinBox();
        min->setDecimals(6);
        max->setDecimals(6);
        fixed->setDecimals(6);
        sigma->setDecimals(6);
        min->setRange(-1e9, 1e9);
        max->setRange(-1e9, 1e9);
        fixed->setRange(-1e9, 1e9);
        sigma->setRange(0, 1e9);

        QComboBox *cb = new QComboBox();
        connect(cb, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, [=](int idx){
            min->setEnabled(idx==1);
            max->setEnabled(idx==1);
            fixed->setEnabled(idx==2);
            sigma->setEnabled(idx == 1 && i < model.nNormalAdjustableParams);
            if ( idx == 4 ) { // Reset
                min->setValue(p.min);
                max->setValue(p.max);
                fixed->setValue(p.initial);
                sigma->setValue(p.sigma);
                cb->setCurrentIndex(0);
            }
        });
        cb->addItems({"Original", "Range", "Fixed", "Target", "Reset"});

        ui->constraints->setCellWidget(i, 0, cb);
        ui->constraints->setCellWidget(i, 1, sigma);
        ui->constraints->setCellWidget(i, 2, fixed);
        ui->constraints->setCellWidget(i, 3, min);
        ui->constraints->setCellWidget(i, 4, max);
    }
    ui->constraints->setVerticalHeaderLabels(labels);
    ui->constraints->setColumnWidth(0, 75);

    connect(ui->sigma_add_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ )
            if ( !session.project.model().adjustableParams[i].multiplicative )
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 1))->setValue(ui->sigma_add->value());
    });
    connect(ui->sigma_mul_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ )
            if ( session.project.model().adjustableParams[i].multiplicative )
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 1))->setValue(ui->sigma_mul->value());
    });
    connect(ui->range_add_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
            if ( !session.project.model().adjustableParams[i].multiplicative ) {
                double baseVal = qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->value();
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(baseVal - ui->range_add->value());
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->setValue(baseVal + ui->range_add->value());
            }
        }
    });
    connect(ui->range_mul_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
            if ( session.project.model().adjustableParams[i].multiplicative ) {
                double baseVal = qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->value();
                double delta = ui->range_mul->value();
                if ( delta >= 1 ) {
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(baseVal / delta);
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->setValue(baseVal * delta);
                } else {
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(baseVal - baseVal * delta);
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->setValue(baseVal + baseVal * delta);
                }
            }
        }
    });
    connect(ui->constraints_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ )
            qobject_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->setCurrentIndex(ui->constraints_all->currentIndex());
    });

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
    ui->method->setCurrentIndex(p.useDE ? 1 : 0);
    ui->selectivity->setCurrentIndex(p.mutationSelectivity);

    if ( p.obsSource == Wavegen::cluster_action.toStdString() )
        ui->obsSource->setCurrentIndex(1);
    else if ( p.obsSource == Wavegen::bubble_action.toStdString() )
        ui->obsSource->setCurrentIndex(2);
    else
        ui->obsSource->setCurrentIndex(0);
    ui->chunkDuration->setValue(p.chunkDuration);

    ui->cl_nStims->setValue(p.cl_nStims);
    ui->cl_nSelect->setValue(p.cl_nSelect);
    ui->SDF_size->setValue(p.SDF_size);
    ui->SDF_decay->setValue(p.SDF_decay);
    ui->spike_threshold->setValue(p.spike_threshold);
    ui->cl_validation_interval->setValue(p.cl_validation_interval);

    ui->DE_decay->setValue(p.DE_decay);

    for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
        static_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->setCurrentIndex(p.constraints[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 1))->setValue(p.sigma[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->setValue(p.fixedValue[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(p.min[i]);
        static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->setValue(p.max[i]);
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
    p.useDE = ui->method->currentIndex() == 1;
    p.mutationSelectivity = ui->selectivity->currentIndex();

    switch ( ui->obsSource->currentIndex() ) {
    case 1: p.obsSource = Wavegen::cluster_action.toStdString(); break;
    case 2: p.obsSource = Wavegen::bubble_action.toStdString(); break;
    case 0:
    default: p.obsSource = "-"; break;
    }
    p.chunkDuration = ui->chunkDuration->value();

    p.cl_nStims = ui->cl_nStims->value();
    p.cl_nSelect = ui->cl_nSelect->value();
    p.SDF_size = ui->SDF_size->value();
    p.SDF_decay = ui->SDF_decay->value();
    p.spike_threshold = ui->spike_threshold->value();
    p.cl_validation_interval = ui->cl_validation_interval->value();

    p.DE_decay = ui->DE_decay->value();

    for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
        p.constraints.push_back(static_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->currentIndex());
        p.sigma.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 1))->value());
        p.fixedValue.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->value());
        p.min.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->value());
        p.max.push_back(static_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->value());
    }

    p.useLikelihood = false;
    p.useClustering = false;

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
