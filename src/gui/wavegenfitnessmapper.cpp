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


#include "wavegenfitnessmapper.h"
#include "ui_wavegenfitnessmapper.h"
#include <QButtonGroup>
#include <QDoubleSpinBox>
#include "wavesource.h"

WavegenFitnessMapper::WavegenFitnessMapper(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WavegenFitnessMapper),
    session(session)
{
    ui->setupUi(this);
    initPlot();

    ui->dimensions->setColumnWidth(0, 25);
    ui->dimensions->setColumnWidth(1, 25);

    updateCombo();
    updateDimensions();
    for ( const AdjustableParam &p : session.project.model().adjustableParams )
        ui->targetParam->addItem(QString::fromStdString(p.name));

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateCombo()));
    connect(ui->combo, SIGNAL(currentIndexChanged(int)), this, SLOT(updateDimensions()));
    connect(ui->btnDraw, SIGNAL(clicked(bool)), this, SLOT(replot()));
    connect(ui->targetParam, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
}

WavegenFitnessMapper::~WavegenFitnessMapper()
{
    delete ui;
    delete colorMap;
}

void WavegenFitnessMapper::initPlot()
{
    ui->plot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
    ui->plot->axisRect()->setupFullAxesBox(true);

    colorMap = new QCPColorMap(ui->plot->xAxis, ui->plot->yAxis);
    colorMap->setInterpolate(false);

    // add a color scale:
    QCPColorScale *colorScale = new QCPColorScale(ui->plot);
    ui->plot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    colorScale->axis()->setLabel("Fitness");

    // Make a heat gradient with blue for exactly 0, to disambiguate between "no data" and "really low fitness"
    QCPColorGradient foo(QCPColorGradient::gpHot);
    foo.setColorStopAt(0, QColor("blue"));
    foo.setColorStopAt(__DBL_MIN__, QColor("black"));
    colorMap->setGradient(foo);

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    QCPMarginGroup *marginGroup = new QCPMarginGroup(ui->plot);
    ui->plot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
}

void WavegenFitnessMapper::updateCombo()
{
    int nArchives = session.wavegen().archives().size();
    int nSelections = session.wavesets().selections().size();
    for ( int i = 0; i < nArchives + nSelections; i++ ) {
        WaveSource src(session,
                       i < nArchives ? WaveSource::Archive : WaveSource::Selection,
                       i < nArchives ? i : (i - nArchives));
        if ( i == ui->combo->count() || !(ui->combo->itemData(i).value<WaveSource>() == src) )
            ui->combo->insertItem(i, src.prettyName(), QVariant::fromValue(src));
    }
}

void WavegenFitnessMapper::updateDimensions()
{
    if ( ui->combo->count() == 0 || ui->combo->currentIndex() < 0 )
        return;

    WaveSource src = ui->combo->currentData().value<WaveSource>();
    int i = 0, n = src.archive()->searchd(session).mapeDimensions.size() - 1;
    groupx = new QButtonGroup(this);
    groupy = new QButtonGroup(this);
    mins.resize(n);
    maxes.resize(n);
    collapse.resize(n);
    pareto.resize(n);
    tolerance.resize(n);
    ui->dimensions->setRowCount(n);
    QStringList labels;
    for ( MAPEDimension d : src.archive()->searchd(session).mapeDimensions ) {
        if ( d.func == MAPEDimension::Func::EE_ParamIndex )
            continue;
        labels << QString::fromStdString(toString(d.func));

        QRadioButton *x = new QRadioButton();
        groupx->addButton(x, i);
        ui->dimensions->setCellWidget(i, 0, x);

        QRadioButton *y = new QRadioButton();
        groupy->addButton(y, i);
        ui->dimensions->setCellWidget(i, 1, y);

        QCheckBox *c = new QCheckBox();
        ui->dimensions->setCellWidget(i, 4, c);
        collapse[i] = c;

        QComboBox *cb = new QComboBox();
        cb->addItem("max");
        cb->addItem("min");
        ui->dimensions->setCellWidget(i, 5, cb);
        pareto[i] = cb;

        QSpinBox *tol = new QSpinBox();
        tol->setSuffix(" bins");
        ui->dimensions->setCellWidget(i, 6, tol);
        tolerance[i] = tol;

        double step = (d.max - d.min) / (d.multiplier(src.archive()->precision) * d.resolution);
        int decimals = 1 - log10(step);

        QDoubleSpinBox *min = new QDoubleSpinBox();
        min->setDecimals(decimals);
        min->setSingleStep(step);
        min->setRange(d.min, d.max);
        min->setValue(d.min);
        ui->dimensions->setCellWidget(i, 2, min);
        mins[i] = min;

        QDoubleSpinBox *max = new QDoubleSpinBox();
        max->setDecimals(decimals);
        max->setSingleStep(step);
        max->setRange(d.min, d.max);
        max->setValue(d.max);
        ui->dimensions->setCellWidget(i, 3, max);
        maxes[i] = max;

        if ( src.selection() ) {
            // Set selection parameters; ensure that min/max aren't rounded the wrong way by placing the values in the center of the bin
            min->setValue(src.selection()->rmin(i+1) + step/2);
            max->setValue(src.selection()->rmax(i+1) + step/2);
            c->setChecked(src.selection()->ranges.at(i+1).collapse);
            cb->setCurrentIndex(!src.selection()->paretoMaximise.at(i+1));
            tol->setValue(src.selection()->paretoTolerance.at(i+1));
        }
        if ( !src.selection() || src.selection()->width(i+1) > 1 ) {
            if ( groupx->checkedId() < 0 )
                x->setChecked(true);
            else if ( groupy->checkedId() < 0 )
                y->setChecked(true);
        }

        ++i;
    }
    ui->dimensions->setVerticalHeaderLabels(labels);

    ui->minFitness->setValue(src.selection() ? src.selection()->minFitness : 0);
    ui->paretoFront->setChecked(src.selection() ? src.selection()->paretoFront : false);
    ui->fitnessTolerance->setValue(src.selection() ? src.selection()->paretoFitnessTol : 0);
    ui->targetParam->setCurrentIndex((src.selection() && src.selection()->width(0)==1) ? src.selection()->ranges[0].min+1 : 0);

    int axis = ui->limitAxis->currentIndex();
    for ( int i = ui->limitAxis->count() - 1; i > 0; i-- )
        ui->limitAxis->removeItem(i);
    ui->limitAxis->addItems(labels);
    ui->limitAxis->setCurrentIndex(axis);

    replot();
}

bool WavegenFitnessMapper::select(bool flattenToPlot)
{
    if ( flattenToPlot ) {
        // Recurse without artificial flattening to set limit to true nFinal, and if checked to limit over the full dimensionality
        // This updates limits and unchecks limitCheck, see below.
        select(false);
    }

    WaveSource src = ui->combo->currentData().value<WaveSource>();
    if ( src.type == WaveSource::Selection ) {
        selection.reset(new WavegenSelection(*src.selection()));
    } else {
        selection.reset(new WavegenSelection(session, src.idx));
    }

    if ( ui->targetParam->currentIndex() > 0 ) {
        size_t targetParam = ui->targetParam->currentIndex()-1;
        selection->limit(0, targetParam, targetParam, false);
    } else {
        selection->limit(0, 0, session.project.model().adjustableParams.size(), flattenToPlot);
    }
    for ( int i = 0; i < ui->dimensions->rowCount(); i++ ) {
        bool flatten = collapse[i]->isChecked();
        if ( flattenToPlot )
            flatten |= !(groupx->checkedId() == i || groupy->checkedId() == i);
        selection->limit(i+1, mins[i]->value(), maxes[i]->value(), flatten);
        selection->paretoMaximise[i+1] = !pareto[i]->currentIndex();
        selection->paretoTolerance[i+1] = tolerance[i]->value();
    }
    selection->minFitness = ui->minFitness->value();
    selection->paretoFront = ui->paretoFront->isChecked();
    selection->paretoFitnessTol = ui->fitnessTolerance->value();
    selection->finalise();

    if ( ui->limitCheck->isChecked() ) {
        ui->limitCheck->setChecked(false);
        bool reFinalise = false;

        if ( ui->limitAxis->currentIndex() == 0 ) { // Fitness; ignore direction
            selection->minFitness = selection->getFitnessSizeLimit(ui->limit->value());
            reFinalise = ui->minFitness->value() != selection->minFitness;
            ui->minFitness->setValue(selection->minFitness);
        } else { // MAPE Dimension
            size_t dim = ui->limitAxis->currentIndex();
            bool descending = ui->limitDirection->currentIndex()==0;
            size_t lim = selection->getSizeLimit(ui->limit->value(), dim, descending);
            if ( descending ) {
                reFinalise = lim != selection->ranges[dim].min;
                selection->ranges[dim].min = lim;
                mins[dim]->setValue(selection->rmin(dim));
            } else {
                reFinalise = lim != selection->ranges[dim].max;
                selection->ranges[dim].max = lim;
                maxes[dim]->setValue(selection->rmax(dim));
            }
        }

        if ( reFinalise )
            selection->finalise();
    }
    if ( !flattenToPlot )
        ui->limit->setValue(selection->nFinal);

    return true;
}

void WavegenFitnessMapper::replot()
{
    int x = groupx->checkedId() + 1, y = groupy->checkedId() + 1;
    if ( x == y || x < 1 || y < 1 )
        return;

    if ( !select(true) )
        return;

    MAPEDimension dimx = selection->archive().searchd(session).mapeDimensions[x];
    MAPEDimension dimy = selection->archive().searchd(session).mapeDimensions[y];
    int precision = selection->archive().precision;

    // Set up axes
    ui->plot->xAxis->setLabel(QString::fromStdString(toString(dimx.func)));
    ui->plot->yAxis->setLabel(QString::fromStdString(toString(dimy.func)));

    // set up the QCPColorMap:
    colorMap->data()->clear();
    colorMap->data()->setSize(dimx.resolution * dimx.multiplier(precision), dimy.resolution * dimy.multiplier(precision));
    if ( selection->ranges.at(x).collapse )
        colorMap->data()->setKeySize(1);
    if ( selection->ranges.at(y).collapse )
        colorMap->data()->setValueSize(1);
    colorMap->data()->setRange(QCPRange(dimx.min, dimx.max), QCPRange(dimy.min, dimy.max));
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:
    // Note, plot area spans the full dimensional range, but only selection is assigned
    for ( std::vector<size_t> idx(selection->archive().searchd(session).mapeDimensions.size()); idx[x] < selection->width(x); ++idx[x] ) {
        for ( idx[y] = 0; idx[y] < selection->width(y); ++idx[y] ) {
            bool ok;
            auto it = selection->data_relative(idx, &ok);
            colorMap->data()->setCell(
                idx[x] + selection->ranges.at(x).min,
                idx[y] + selection->ranges.at(y).min,
                ok ? it->fitness : 0);
        }
    }

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();
    if ( selection->minFitness > 0 )
        colorMap->setDataRange(QCPRange(selection->minFitness, colorMap->dataRange().upper));

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    ui->plot->rescaleAxes();
    ui->plot->replot();
}

void WavegenFitnessMapper::on_btnAdd_clicked()
{
    if ( !select(false) )
        return;
    WavesetCreator &creator = session.wavesets();
    session.queue(creator.actorName(), creator.actionSelect, "", selection.release());
}

void WavegenFitnessMapper::on_readMinFitness_clicked()
{
    ui->minFitness->setValue(colorMap->dataRange().lower);
}

void WavegenFitnessMapper::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->plot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), ui->combo->currentText());
}

void WavegenFitnessMapper::on_readMaxFitness_clicked()
{
    ui->fitnessTolerance->setValue(colorMap->dataRange().upper * ui->tolFitnessFraction->value());
}

void WavegenFitnessMapper::on_deltabar_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    std::ofstream os(file.toStdString(), std::ios_base::out | std::ios_base::trunc);
    for ( double dbar : ui->combo->currentData().value<WaveSource>().archive()->deltabar )
        os << dbar << '\n';
}
