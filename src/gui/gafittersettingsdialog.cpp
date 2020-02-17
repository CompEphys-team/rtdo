/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "gafittersettingsdialog.h"
#include "ui_gafittersettingsdialog.h"
#include <functional>
#include "populationsaver.h"
#include "util.h"

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
    } else {
        ui->buttonBox->setStandardButtons(QDialogButtonBox::Close | QDialogButtonBox::Apply);
        ui->buttonBox->button(QDialogButtonBox::Apply)->setText("Reapply");
    }
    connect(this, SIGNAL(apply(GAFitterSettings)), &session, SLOT(setGAFitterSettings(GAFitterSettings)));

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

        QTableWidgetItem *relativeDisplay = new QTableWidgetItem("");
        std::function<void(void)> setRelativeRange;
        if ( p.multiplicative ) {
            setRelativeRange = [=](){
                double lo = min->value(), hi = max->value(), c = fixed->value();
                if ( c < 0 )
                    std::swap(lo, hi);
                double ldelta = lo/c, hdelta = hi/c;
                if ( (1-ldelta)/(hdelta-1) > 0.95 && (1-ldelta)/(hdelta-1) < 1.05 )
                    relativeDisplay->setText(QString::number((hdelta-ldelta)/2));
                else if ( ldelta*hdelta > 0.95 && ldelta*hdelta < 1.05 )
                    relativeDisplay->setText(QString::number((1/ldelta+hdelta)/2));
                else
                    relativeDisplay->setText("");
            };
        } else {
            setRelativeRange = [=](){
                double lo = min->value(), hi = max->value(), c = fixed->value();
                double ldelta = c-lo, hdelta = hi-c;
                if ( ldelta/hdelta > 0.95 && ldelta/hdelta < 1.05 )
                    relativeDisplay->setText(QString::number((ldelta+hdelta)/2));
                else
                    relativeDisplay->setText("");
            };
        }
        connect(min, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), setRelativeRange);
        connect(max, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), setRelativeRange);

        ui->constraints->setCellWidget(i, 0, cb);
        ui->constraints->setCellWidget(i, 1, sigma);
        ui->constraints->setCellWidget(i, 2, fixed);
        ui->constraints->setCellWidget(i, 3, min);
        ui->constraints->setCellWidget(i, 4, max);
        ui->constraints->setItem(i, 5, relativeDisplay);
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
            if ( qobject_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->currentIndex() == 1 && !session.project.model().adjustableParams[i].multiplicative ) {
                double baseVal = qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->value();
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(baseVal - ui->range_add->value());
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->setValue(baseVal + ui->range_add->value());
            }
        }
    });
    connect(ui->range_mul_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ ) {
            if ( qobject_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->currentIndex() == 1 && session.project.model().adjustableParams[i].multiplicative ) {
                double baseVal = qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->value();
                double delta = ui->range_mul->value();
                int small = baseVal>0 ? 3 : 4;
                int large = baseVal>0 ? 4 : 3;
                if ( delta >= 1 ) {
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, small))->setValue(baseVal / delta);
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, large))->setValue(baseVal * delta);
                } else {
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, small))->setValue(baseVal - baseVal * delta);
                    qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, large))->setValue(baseVal + baseVal * delta);
                }
            }
        }
    });
    connect(ui->constraints_set, &QPushButton::clicked, [=](){
        for ( int i = 0; i < ui->constraints->rowCount(); i++ )
            qobject_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->setCurrentIndex(ui->constraints_all->currentIndex());
    });

    auto update_dmap_hi = [=](){
        ui->cl_dmap_hi->setValue(session.project.universal().cl_dmap_hi(ui->cl_dmap_lo->value(), ui->cl_dmap_step->value()));
    };
    connect(ui->cl_dmap_lo, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), update_dmap_hi);
    connect(ui->cl_dmap_step, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), update_dmap_hi);

    auto addfits = [=](){
        for ( size_t i = ui->fits->count(); i < session.gaFitter().results().size(); i++ )
            ui->fits->addItem(QString("Fit %1 (%2)").arg(i).arg(session.gaFitter().results().at(i).resultIndex, 4, 10, QChar('0')));
    };
    addfits();
    connect(&session.gaFitter(), &GAFitter::done, this, addfits);
    connect(ui->fit_range_set, &QPushButton::clicked, [=](){
        if ( ui->fits->currentIndex() < 0 )
            return;
        UniversalLibrary lib(session.project, false, true);
        const GAFitter::Output &fit = session.gaFitter().results().at(ui->fits->currentIndex());
        QFile basefile(session.resultFilePath(fit.resultIndex));
        PopLoader loader(basefile, lib);
        loader.load(fit.epochs-1, lib);
        std::vector<double> values(lib.NMODELS);
        int k = ui->fit_range->value();
        std::vector<double> quantiles({k*0.01, 1-k*0.01});
        for ( size_t i = 0; i < lib.adjustableParams.size(); i++ ) {
            if ( k > 0 && qobject_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->currentIndex() == 1 ) {
                for ( size_t j = 0; j < lib.NMODELS; j++ )
                    values[j] = lib.adjustableParams[i][j];
                std::vector<double> q = Quantile(values, quantiles);
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 3))->setValue(std::min(q[0], q[1]));
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 4))->setValue(std::max(q[0], q[1]));
            } else if ( k == 0 && qobject_cast<QComboBox*>(ui->constraints->cellWidget(i, 0))->currentIndex() == 2 ) {
                qobject_cast<QDoubleSpinBox*>(ui->constraints->cellWidget(i, 2))->setValue(fit.final ? fit.finalParams[i] : fit.params[fit.epochs-1][i]);
            }
        }
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
    ui->num_populations->setValue(p.num_populations);

    if ( p.obsSource == Wavegen::cluster_action.toStdString() )
        ui->obsSource->setCurrentIndex(1);
    else if ( p.obsSource == Wavegen::bubble_action.toStdString() )
        ui->obsSource->setCurrentIndex(2);
    else if ( p.obsSource == "random" )
        ui->obsSource->setCurrentIndex(3);
    else
        ui->obsSource->setCurrentIndex(0);
    ui->chunkDuration->setValue(p.chunkDuration);

    ui->cl_nStims->setValue(p.cl_nStims);
    ui->cl_nSelect->setValue(p.cl_nSelect);
    ui->cl_validation_interval->setValue(p.cl_validation_interval);
    ui->cl_trace_weight->setValue(p.cl.err_weight_trace);
    ui->cl_trace_K->setValue(p.cl.Kfilter);
    ui->cl_trace_K2->setValue(p.cl.Kfilter2);
    ui->cl_sdf_weight->setValue(p.cl.err_weight_sdf);
    ui->cl_sdf_threshold->setValue(p.cl.spike_threshold);
    ui->cl_sdf_tau->setValue(p.cl.sdf_tau);
    ui->cl_dmap_weight->setValue(p.cl.err_weight_dmap);
    ui->cl_tDelay->setValue(p.cl.tDelay);
    ui->cl_dmap_lo->setValue(p.cl.dmap_low);
    ui->cl_dmap_step->setValue(p.cl.dmap_step);
    ui->cl_dmap_sigma->setValue(p.cl.dmap_sigma);

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
    p.num_populations = ui->num_populations->value();

    switch ( ui->obsSource->currentIndex() ) {
    case 1: p.obsSource = Wavegen::cluster_action.toStdString(); break;
    case 2: p.obsSource = Wavegen::bubble_action.toStdString(); break;
    case 3: p.obsSource = "random"; break;
    case 0:
    default: p.obsSource = "-"; break;
    }
    p.chunkDuration = ui->chunkDuration->value();

    p.cl_nStims = ui->cl_nStims->value();
    p.cl_nSelect = ui->cl_nSelect->value();
    p.cl_validation_interval = ui->cl_validation_interval->value();
    p.cl.err_weight_trace = ui->cl_trace_weight->value();
    p.cl.Kfilter = ui->cl_trace_K->value();
    p.cl.Kfilter2 = ui->cl_trace_K2->value();
    p.cl.err_weight_sdf = ui->cl_sdf_weight->value();
    p.cl.spike_threshold = ui->cl_sdf_threshold->value();
    p.cl.sdf_tau = ui->cl_sdf_tau->value();
    p.cl.err_weight_dmap = ui->cl_dmap_weight->value();
    p.cl.tDelay = ui->cl_tDelay->value();
    p.cl.dmap_low = ui->cl_dmap_lo->value();
    p.cl.dmap_step = ui->cl_dmap_step->value();
    p.cl.dmap_sigma = ui->cl_dmap_sigma->value();

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
