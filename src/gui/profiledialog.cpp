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


#include "profiledialog.h"
#include "ui_profiledialog.h"
#include "session.h"
#include <QSpinBox>
#include <QDoubleSpinBox>
#include "wavesource.h"
#include <QMessageBox>

ProfileDialog::ProfileDialog(Session &s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ProfileDialog),
    session(s)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateCombo()));
    connect(&session.profiler(), SIGNAL(done()), this, SLOT(updatePresets()));

    QStringList labels;
    int rows = session.project.model().adjustableParams.size();
    ui->table->setRowCount(rows);
    ns.resize(rows);
    mins.resize(rows);
    maxes.resize(rows);
    int i = 0;
    for ( AdjustableParam const& p : session.project.model().adjustableParams ) {
        labels << QString::fromStdString(p.name);

        QSpinBox *n = new QSpinBox();
        n->setRange(0, session.project.universal().NMODELS);
        n->setSingleStep(10);
        ui->table->setCellWidget(i, 0, n);
        ns[i] = n;

        int decimals = 4 - log10(fabs(p.min == 0 ? p.initial : p.min));
        QDoubleSpinBox *min = new QDoubleSpinBox();
        min->setDecimals(decimals);
        min->setRange(p.min, p.max);
        min->setValue(p.min);
        ui->table->setCellWidget(i, 1, min);
        mins[i] = min;

        QDoubleSpinBox *max = new QDoubleSpinBox();
        max->setDecimals(decimals);
        max->setRange(p.min, p.max);
        max->setValue(p.max);
        ui->table->setCellWidget(i, 2, max);
        maxes[i] = max;

        connect(n, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=](int v) {
            if ( v == 0 ) {
                min->setValue(p.min);
                max->setValue(p.max);
            } else if ( v == 1 ) {
                min->setValue(p.initial);
                ui->table->model()->setData(ui->table->model()->index(i, 1), true, Qt::UserRole); // Set "clean" flag
            }
            min->setEnabled(v > 0);
            max->setEnabled(v > 1);

            double perms = 1;
            for ( QSpinBox *box : ns ) {
                if ( box && box->value() == 0 )
                    perms *= session.project.universal().NMODELS;
                else if ( box )
                    perms *= box->value();
            }
            ui->permutationInfo->setText(QString("Total permutations per waveform: %1").arg(perms));
        });
        connect(min, static_cast<void(QDoubleSpinBox::*)(double)>(&QDoubleSpinBox::valueChanged), [=](double) {
            ui->table->model()->setData(ui->table->model()->index(i, 1), false, Qt::UserRole); // Set "dirty" flag
        });

        n->setValue(1);

        ++i;
    }
    ui->table->setVerticalHeaderLabels(labels);
    ui->table->setColumnWidth(0, 50);

    updateCombo();
    updatePresets();
}

ProfileDialog::~ProfileDialog()
{
    delete ui;
}

void ProfileDialog::updateCombo()
{
    int currentIdx = ui->cbSelection->currentIndex();
    WaveSource currentData = ui->cbSelection->currentData().value<WaveSource>();
    ui->cbSelection->clear();
    std::vector<WaveSource> sources = session.wavesets().sources();
    for ( const WaveSource &src : sources ) {
        ui->cbSelection->addItem(src.prettyName(), QVariant::fromValue(src));
    }

    if ( currentIdx < 0 )
        return;

    ui->cbSelection->setCurrentIndex(currentData.index());
}

void ProfileDialog::updatePresets()
{
    if ( ui->preset->count() == 0 ) {
        ui->preset->addItem("Target only, full range");
        ui->preset->addItem("Target only, close range");
        ui->preset->addItem("Target full range, others close range");
        ui->preset->addItem("Target close range, others close range");
        ui->preset->addItem("Target full range, others full range");
        ui->preset->addItem("Target close range, others full range");
    }

    int currentIdx = ui->preset->currentIndex();
    while ( ui->preset->count() > nHardPresets )
        ui->preset->removeItem(nHardPresets);

    for ( ErrorProfile const& ep : session.profiler().profiles() )
        ui->preset->addItem(ep.prettyName());

    ui->preset->setCurrentIndex(currentIdx);
}

void ProfileDialog::setCloseRange(int i)
{
    double min, max;
    const AdjustableParam &p = session.project.model().adjustableParams[i];
    if ( p.multiplicative ) {
        max = p.initial * pow(1.0 + p.sigma, ui->presetSig->value());
        min = p.initial * pow(1.0 - p.sigma, ui->presetSig->value());
    } else {
        max = p.initial + p.sigma*ui->presetSig->value();
        min = p.initial - p.sigma*ui->presetSig->value();
    }
    if ( max > p.max )
        max = p.max;
    if ( min < p.min )
        min = p.min;
    maxes[i]->setValue(max);
    mins[i]->setValue(min);
}

void ProfileDialog::on_btnPreset_clicked()
{
    if ( ui->preset->currentIndex() < 0 || ui->cbSelection->currentIndex() < 0 )
        return;
    WaveSource src = ui->cbSelection->currentData().value<WaveSource>();
    int preset = ui->preset->currentIndex();

    if ( preset < nHardPresets ) {
        int targetParam = -1;
        if ( src.selection() && src.selection()->width(0) == 1 ) {
            targetParam = src.selection()->ranges[0].min;
            if ( ns[targetParam]->value() == 0 ) // Target: Set full range via valueChanged signal
                ns[targetParam]->valueChanged(0);
            else
                ns[targetParam]->setValue(0);
            if ( preset == 1 || preset == 3 || preset == 5 ) { // Target: set close range
                setCloseRange(targetParam);
            }
            if ( preset == 0 || preset == 1 ) { // Target only
                for ( int i = 0; i < int(ns.size()); i++ ) {
                    if ( i != targetParam ) {
                        ns[i]->setValue(1);
                    }
                }
            }
        } else {
            // Convert non-existant target to other
            if ( preset == 0 )  preset = 4;
            if ( preset == 1 )  preset = 2;
        }
        if ( preset == 2 || preset == 3 ) { // Others: set close range
            for ( int i = 0; i < int(ns.size()); i++ ) {
                if ( i != targetParam ) {
                    ns[i]->setValue(ui->presetN->value());
                    setCloseRange(i);
                }
            }
        } else if ( preset == 4 || preset == 5 ) { // Others: set full range
            for ( int i = 0; i < int(ns.size()); i++ ) {
                if ( i != targetParam ) {
                    ns[i]->setValue(ui->presetN->value());
                    mins[i]->setValue(mins[i]->minimum());
                    maxes[i]->setValue(maxes[i]->maximum());
                }
            }
        }
    } else {
        const ErrorProfile &ep = session.profiler().profiles().at(preset - nHardPresets);
        for ( size_t i = 0; i < ep.permutations().size(); i++ ) {
            const ErrorProfile::Permutation &perm = ep.permutations().at(i);
            if ( perm.n == session.project.universal().NMODELS ) {
                ns[i]->setValue(0);
            } else {
                ns[i]->setValue(perm.n);
            }
            if ( perm.fixed ) {
                mins[i]->setValue(perm.value);
            } else if ( perm.n == 1 ) {
                mins[i]->setValue(session.project.model().adjustableParams[i].initial);
            } else {
                mins[i]->setValue(perm.min);
                maxes[i]->setValue(perm.max);
            }
        }
    }
}

void ProfileDialog::on_btnStart_clicked()
{
    if ( ui->cbSelection->currentIndex() < 0 )
        return;

    ErrorProfile profile(session);
    profile.setSource(ui->cbSelection->currentData().value<WaveSource>());

    for ( size_t i = 0; i < session.project.model().adjustableParams.size(); i++ ) {
        ErrorProfile::Permutation perm;
        perm.n = ns[i]->value();
        if ( perm.n == 1 && !ui->table->model()->data(ui->table->model()->index(i, 1), Qt::UserRole).toBool() ) {
            perm.fixed = true;
            perm.value = mins[i]->value();
        }
        perm.min = mins[i]->value();
        perm.max = maxes[i]->value();
        profile.setPermutation(i, perm);
    }

    session.profiler().queueProfile(std::move(profile));
}

void ProfileDialog::on_btnAbort_clicked()
{
    session.abort();
}
