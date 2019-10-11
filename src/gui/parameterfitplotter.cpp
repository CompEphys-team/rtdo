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


#include "parameterfitplotter.h"
#include "ui_parameterfitplotter.h"
#include "colorbutton.h"
#include <QTimer>

ParameterFitPlotter::ParameterFitPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ParameterFitPlotter),
    summarising(false)
{
    ui->setupUi(this);
    connect(ui->slider, SIGNAL(valueChanged(int)), this, SLOT(resizePanel()));
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(clearPlotLayout()));
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(buildPlotLayout()));
    connect(ui->legend, SIGNAL(stateChanged(int)), this, SLOT(replot()));
    connect(ui->filter, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=](int){
        replot();
    });
    connect(ui->param, &QCheckBox::stateChanged, [=](int state) {
        bool on = state == Qt::Checked;
        for ( QCPAxisRect *ar : axRects )
            for ( QCPGraph *g : ar->axis(QCPAxis::atLeft, summarising?1:0)->graphs() )
                g->setVisible(on);
        setGridAndAxVisibility();
        ui->panel->layer("target")->setVisible(on && !summarising);
        ui->panel->replot();
    });
    connect(ui->error, &QCheckBox::stateChanged, [=](int state) {
        bool on = state == Qt::Checked;
        for ( QCPAxisRect *ar : axRects ) {
            for ( QCPGraph *g : ar->axis(QCPAxis::atRight)->graphs() )
                g->setVisible(on);
            ar->axis(QCPAxis::atRight)->setVisible(on);
        }
        ui->panel->replot();
    });
    connect(ui->opacity, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=](int op){
        double opacity = op/100.;
        QList<QCPLayerable*> normalGraphs;
        normalGraphs.append(ui->panel->layer("main")->children());
        normalGraphs.append(ui->panel->layer("mean")->children());
        normalGraphs.append(ui->panel->layer("median")->children());
        normalGraphs.append(ui->panel->layer("max")->children());
        for ( QCPLayerable *l : normalGraphs ) {
            QCPGraph *g = qobject_cast<QCPGraph*>(l);
            if ( !g ) continue;
            QPen pen = g->pen();
            QColor col = pen.color();
            col.setAlphaF(opacity);
            pen.setColor(col);
            g->setPen(pen);
        }
        for ( QCPLayerable *l : ui->panel->layer("sem")->children() ) {
            QCPGraph *g = qobject_cast<QCPGraph*>(l);
            if ( !g ) continue;
            QColor col = g->pen().color();
            col.setAlphaF(0.4*opacity);
            g->setPen(QPen(col));
            col.setAlphaF(0.2*opacity);
            g->setBrush(QBrush(col));
        }
        ui->panel->replot();
    });
    connect(ui->mean, &QCheckBox::toggled, [=](bool on){
        ui->panel->layer("mean")->setVisible(on);
        ui->panel->replot();
    });
    connect(ui->SEM, &QCheckBox::toggled, [=](bool on){
        ui->panel->layer("sem")->setVisible(on);
        ui->panel->replot();
    });
    connect(ui->median, &QCheckBox::toggled, [=](bool on){
        ui->panel->layer("median")->setVisible(on);
        ui->panel->replot();
    });
    connect(ui->max, &QCheckBox::toggled, [=](bool on){
        ui->panel->layer("max")->setVisible(on);
        ui->panel->replot();
    });

    ui->panel->addLayer("mean");
    ui->panel->layer("mean")->setVisible(ui->mean->isChecked());
    ui->panel->addLayer("sem");
    ui->panel->layer("sem")->setVisible(ui->SEM->isChecked());
    ui->panel->addLayer("median");
    ui->panel->layer("median")->setVisible(ui->median->isChecked());
    ui->panel->addLayer("max");
    ui->panel->layer("max")->setVisible(ui->max->isChecked());
    ui->panel->addLayer("target");
    ui->panel->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
    connect(ui->panel, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->panel->selectedAxes();
        if ( axes.isEmpty() ) {
            for ( QCPAxisRect *ar : axRects ) {
                ar->setRangeZoomAxes(ar->axes());
                ar->setRangeDragAxes(ar->axes());
            }
        } else {
            QCPAxis *ax = axes.first(); // Note, no multiselect interaction; thus, list can only be one element long
            int idx = 0;
            while ( ax != ax->axisRect()->axis(ax->axisType(), idx) )
                ++idx;
            for ( QCPAxisRect *ar : axRects ) {
                QCPAxis *axis = ar->axis(ax->axisType(), idx);
                axis->setSelectedParts(QCPAxis::spAxis);
                ar->setRangeDragAxes({axis});
                ar->setRangeZoomAxes({axis});
            }
        }
    });
    connect(ui->panel, &QCustomPlot::axisDoubleClick, this, [=](QCPAxis *axis, QCPAxis::SelectablePart, QMouseEvent*) {
        QCPAxisRect *ar = axis->axisRect();
        if ( axis == ar->axis(QCPAxis::atLeft, 0) ) {
            for ( size_t i = 0; i < axRects.size(); i++ ) {
                if ( axRects[i] == ar ) {
                    const AdjustableParam &p = session->project.model().adjustableParams[i];
                    axis->setRange(p.min, p.max);
                    break;
                }
            }
        } else if ( axis == ar->axis(QCPAxis::atBottom) )
            axis->rescale();
        else if ( axis == ar->axis(QCPAxis::atLeft, 1) )
            axis->setRange(0, 100);
        else if ( !axis->graphs().isEmpty() ) { // atRight
            bool foundRange;
            QCPRange range = axis->graphs().first()->getValueRange(foundRange, QCP::sdBoth, ar->axis(QCPAxis::atBottom)->range());
            if ( foundRange )
                axis->setRange(range);
        }
        ui->panel->replot();
    });
}

ParameterFitPlotter::ParameterFitPlotter(Session &session, QWidget *parent) :
    ParameterFitPlotter(parent)
{
    init(&session, false);
}

ParameterFitPlotter::~ParameterFitPlotter()
{
    delete ui;
}

void ParameterFitPlotter::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    resizePanel();
}

void ParameterFitPlotter::resizePanel()
{
    double height = std::max(1, ui->slider->height() * ui->slider->value() / ui->slider->maximum());
    int nRows = (axRects.size() + ui->columns->value() - 1) / ui->columns->value();
    ui->panel->setFixedHeight(height * nRows);

    int legendWidth = 0;
    if ( ui->panel->plotLayout()->columnCount() > 1 )
        legendWidth = ui->panel->plotLayout()->element(0, 1)->outerRect().width();
    ui->panel->setFixedWidth(ui->scrollArea->childrenRect().width() + legendWidth);

    ui->panel->replot();
}

void ParameterFitPlotter::setGridAndAxVisibility()
{
    bool on = ui->param->isChecked();
    for ( QCPAxisRect *ar : axRects ) {
        ar->axis(QCPAxis::atLeft, 1)->setVisible(on && summarising);
        ar->axis(QCPAxis::atLeft, 0)->setVisible(!(on && summarising));
        ar->axis(QCPAxis::atLeft, 0)->setTicks(on);
        ar->axis(QCPAxis::atLeft, 0)->setTickLabels(on);
        ar->axis(QCPAxis::atLeft, 0)->grid()->setVisible(on);
        ar->axis(QCPAxis::atRight)->grid()->setVisible(!on);
    }
}

void ParameterFitPlotter::init(Session *session, bool enslave)
{
    this->session = session;
    this->enslaved = enslave;

    // Plots
    axRects.resize(session->project.model().adjustableParams.size());
    for ( size_t i = 0; i < axRects.size(); i++ ) {
        const AdjustableParam &p = session->project.model().adjustableParams[i];
        QCPAxisRect *ar = new QCPAxisRect(ui->panel);
        axRects[i] = ar;
        ar->setRangeZoomAxes(ar->axes());
        ar->setRangeDragAxes(ar->axes());

        QCPAxis *xAxis = ar->axis(QCPAxis::atBottom);
        xAxis->setLabel("Epoch");
        xAxis->setRange(0, 1000);
        connect(xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(xRangeChanged(QCPRange)));

        QCPAxis *yAxis = ar->axis(QCPAxis::atLeft, 0);
        yAxis->setLabel(QString::fromStdString(p.name));
        yAxis->setRange(p.min, p.max);
        yAxis->setTicks(ui->param->isChecked());
        yAxis->setTickLabels(ui->param->isChecked());

        QCPAxis *yAxis2 = ar->axis(QCPAxis::atRight);
        yAxis2->setLabel("RMS Error (nA)");
        yAxis2->setVisible(ui->error->isChecked());
        connect(yAxis2, SIGNAL(rangeChanged(QCPRange)), this, SLOT(errorRangeChanged(QCPRange)));

        QCPAxis *yAxis3 = ar->addAxis(QCPAxis::atLeft);
        yAxis3->setLabel(QString("Δ %1 (%%2)").arg(QString::fromStdString(p.name)).arg(p.multiplicative ? "" : " range"));
        yAxis3->setRange(0,100);
        yAxis3->grid()->setVisible(true);
        yAxis3->setVisible(false);
        connect(yAxis3, SIGNAL(rangeChanged(QCPRange)), this, SLOT(percentileRangeChanged(QCPRange)));

        ar->setRangeDragAxes(ar->axes());
        ar->setRangeZoomAxes(ar->axes());

        for ( QCPAxis *ax : ar->axes() ) {
            ax->setLayer("axes");
            ax->grid()->setLayer("grid");
        }
    }
    ui->panel->legend->setVisible(true);
    ui->panel->setAutoAddPlottableToLegend(false);
    ui->panel->legend->setFillOrder(QCPLegend::foColumnsFirst);
    ui->panel->legend->setWrap(2);
    ui->panel->axisRect()->insetLayout()->take(ui->panel->legend);
    ui->columns->setMaximum(axRects.size());
    replot();

    // Enslave to GAFitterWidget
    if ( enslave ) {
        ui->legend->setVisible(false);
        ui->opacity->setVisible(false);
        ui->summary_plot_controls->setVisible(false);
        connect(&session->gaFitter(), &GAFitter::starting, this, &ParameterFitPlotter::clear);
        connect(&session->gaFitter(), &GAFitter::progress, this, &ParameterFitPlotter::progress);
        connect(&session->gaFitter(), &GAFitter::done, this, [=](){
            const GAFitter::Output &o = session->gaFitter().results().back();
            if ( o.final )
                addFinal(o);
        });
        clear();
    }
    QTimer::singleShot(10, this, &ParameterFitPlotter::resizePanel);
}

void ParameterFitPlotter::setData(std::vector<FitInspector::Group> data, bool summarising)
{
    this->data = data;
    this->summarising = summarising;
    if ( isVisible() )
        replot();
}

void ParameterFitPlotter::clearPlotLayout()
{
    QCPLayoutGrid *graphLayout = qobject_cast<QCPLayoutGrid*>(ui->panel->plotLayout()->element(0, 0));
    if ( graphLayout ) {
        for ( QCPAxisRect *ar : axRects ) {
            for ( int i = 0; i < graphLayout->elementCount(); i++ ) {
                if ( graphLayout->elementAt(i) == ar ) {
                    graphLayout->takeAt(i);
                    break;
                }
            }
        }
        if ( ui->panel->plotLayout()->columnCount() > 1 )
            qobject_cast<QCPLayoutGrid*>(ui->panel->plotLayout()->element(0, 1))->take(ui->panel->legend);
    }
    ui->panel->plotLayout()->clear();
}

void ParameterFitPlotter::replot()
{
    ui->panel->clearItems();
    ui->panel->clearGraphs();

    clearPlotLayout();

    ui->panel->plotLayout()->addElement(new QCPAxisRect(ui->panel)); // Prevent debug output from within QCP on adding items

    if ( summarising )
        plotSummary();
    else
        plotIndividual();

    ui->summary_plot_controls->setEnabled(summarising);

    buildPlotLayout();
}

void ParameterFitPlotter::buildPlotLayout()
{
    ui->panel->plotLayout()->clear();

    QCPLayoutGrid *graphLayout = new QCPLayoutGrid();
    ui->panel->plotLayout()->addElement(0, 0, graphLayout);

    ui->panel->legend->setVisible(ui->legend->isChecked());
    if ( ui->legend->isChecked() ) {
        QCPLayoutGrid *legendLayout = new QCPLayoutGrid();
        legendLayout->setMargins(QMargins(5, 5, 5, 5));
        legendLayout->addElement(0, 0, ui->panel->legend);
        legendLayout->addElement(1, 0, new QCPLayoutElement(ui->panel));
        legendLayout->setRowStretchFactor(0, 0.001);
        ui->panel->plotLayout()->addElement(0, 1, legendLayout);
        ui->panel->plotLayout()->setColumnStretchFactor(1, 0.001);
    }

    size_t i = 0;
    int n = ui->columns->value();
    for ( int row = 0; row < std::ceil(double(axRects.size())/n); row++ ) {
        for ( int col = 0; col < n; col++ ) {
            graphLayout->addElement(row, col, axRects[i]);
            if ( ++i >= axRects.size() )
                row = col = axRects.size(); // break
        }
    }
    setGridAndAxVisibility();
    ui->panel->replot(QCustomPlot::rpQueuedRefresh);
    resizePanel();
}

void ParameterFitPlotter::plotIndividual()
{
    if ( data.empty() || data[0].fits.empty() )
        return;

    ui->panel->layer("target")->setVisible(ui->param->isChecked());

    for ( const FitInspector::Fit &f : data[0].fits ) {
        const GAFitter::Output fit = f.fit();
        QVector<double> keys(fit.epochs);
        for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ )
            keys[epoch] = epoch;
        for ( size_t i = 0; i < axRects.size(); i++ ) {
            QCPAxis *xAxis = axRects[i]->axis(QCPAxis::atBottom);
            QCPAxis *yAxis = axRects[i]->axis(QCPAxis::atLeft, 0);
            QCPAxis *yAxis2 = axRects[i]->axis(QCPAxis::atRight);

            QCPItemStraightLine *line = new QCPItemStraightLine(ui->panel);
            QPen pen(f.color);
            pen.setStyle(Qt::DashLine);
            line->setLayer("target");
            line->setPen(pen);
            line->point1->setAxes(xAxis, yAxis);
            line->point2->setAxes(xAxis, yAxis);
            line->point1->setCoords(0, fit.targets[i]);
            line->point2->setCoords(1, fit.targets[i]);
            line->setClipAxisRect(axRects[i]);
            line->setClipToAxisRect(true);

            QVector<double> values(fit.epochs), errors, errKey;
            errors.reserve(fit.epochs);
            errKey.reserve(fit.epochs);
            for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ )
                values[epoch] = fit.params[epoch][i];

            if ( fit.stimSource.type == WaveSource::Deck ) {
                for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ ) {
                    if ( fit.targetStim[epoch] == i ) {
                        errors.push_back(fit.error[epoch]);
                        errKey.push_back(epoch);
                    }
                }
            } else if ( i == 0 ) {
                for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ ) {
                    errors.push_back(fit.error[epoch]);
                    errKey.push_back(epoch);
                }
            }

            QColor col(f.color);
            col.setAlphaF(ui->opacity->value()/100.);
            QCPGraph *graph = addGraph(xAxis, yAxis, col, keys, values, "main", ui->param->isChecked());

            QColor errCol(f.errColor);
            errCol.setAlphaF(ui->opacity->value()/100.);
            QCPGraph *errGraph = addGraph(xAxis, yAxis2, errCol, errKey, errors, "main", ui->error->isChecked());
            errGraph->setLineStyle(QCPGraph::lsStepLeft);

            if ( session->gaFitterSettings(fit.resultIndex).useLikelihood )
                yAxis2->setLabel("-log likelihood");
            else
                yAxis2->setLabel("RMS Error (nA)");

            if ( i == 0 ) {
                graph->setName(f.label);
                errGraph->setName("");
                graph->addToLegend();
                errGraph->addToLegend();
            }
        }

        if ( fit.final )
            addFinal(fit);
    }
}

void ParameterFitPlotter::clear()
{
    // Prepare for enslaved plotting through progress()
    ui->panel->clearGraphs();
    ui->panel->clearItems();
    for ( QCPAxisRect *ar : axRects ) {
        QCPAxis *xAxis = ar->axis(QCPAxis::atBottom);
        QCPAxis *yAxis = ar->axis(QCPAxis::atLeft, 0);
        QCPAxis *yAxis2 = ar->axis(QCPAxis::atRight);

        ui->panel->addGraph(xAxis, yAxis)->setVisible(ui->param->isChecked());

        QCPGraph *errGraph = ui->panel->addGraph(xAxis, yAxis2);
        errGraph->setVisible(ui->error->isChecked());
        errGraph->setLineStyle(QCPGraph::lsStepLeft);
        errGraph->setPen(QPen(Qt::magenta));

        xAxis->moveRange(-xAxis->range().lower);
    }
    ui->panel->replot();
}

void ParameterFitPlotter::progress(quint32 epoch)
{
    GAFitter::Output fit = session->gaFitter().currentResults();
    if ( epoch == 0 ) {
        for ( size_t i = 0; i < axRects.size(); i++ ) {
            QCPItemStraightLine *line = new QCPItemStraightLine(ui->panel);
            line->setLayer("target");
            line->point1->setAxes(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft, 0));
            line->point2->setAxes(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft, 0));
            line->point1->setCoords(0, fit.targets[i]);
            line->point2->setCoords(1, fit.targets[i]);
            line->setClipAxisRect(axRects[i]);
            line->setClipToAxisRect(true);
            if ( session->gaFitterSettings().useLikelihood )
                axRects[i]->axis(QCPAxis::atRight)->setLabel("-log likelihood");
            else
                axRects[i]->axis(QCPAxis::atRight)->setLabel("RMS Error (nA)");
        }
    }
    for ( size_t i = 0; i < axRects.size(); i++ ) {
        axRects[i]->axis(QCPAxis::atLeft, 0)->graphs().first()->addData(epoch, fit.params[epoch][i]);
    }
    int targetPlot = fit.targetStim[epoch];
    if ( fit.stimSource.type != WaveSource::Deck )
        targetPlot = 0;
    axRects[targetPlot]->axis(QCPAxis::atRight)->graphs().first()->addData(epoch, fit.error[epoch]);

    double xUpper = axRects[0]->axis(QCPAxis::atBottom)->range().upper;
    if ( double(epoch-1) <= xUpper && double(epoch) > xUpper) {
        for ( QCPAxisRect *ar : axRects ) {
            ar->axis(QCPAxis::atBottom)->blockSignals(true);
            ar->axis(QCPAxis::atBottom)->moveRange(1);
            ar->axis(QCPAxis::atBottom)->blockSignals(false);
        }
    }

    ui->panel->replot(QCustomPlot::rpQueuedReplot);
}

void ParameterFitPlotter::addFinal(const GAFitter::Output &fit)
{
    for ( size_t i = 0; i < axRects.size(); i++ ) {
        QCPAxis *xAxis = axRects[i]->axis(QCPAxis::atBottom);
        QCPAxis *yAxis = axRects[i]->axis(QCPAxis::atLeft, 0);
        QCPAxis *yAxis2 = axRects[i]->axis(QCPAxis::atRight);
        const QCPGraph *gParam = yAxis->graphs().last();
        const QCPGraph *gErr = yAxis2->graphs().last();

        QCPGraph *g = ui->panel->addGraph(xAxis, yAxis);
        g->setVisible(gParam->visible());
        g->setPen(gParam->pen());
        g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle));
        g->addData(fit.epochs, fit.finalParams[i]);

        g = ui->panel->addGraph(xAxis, yAxis2);
        g->setVisible(gErr->visible());
        g->setPen(gErr->pen());
        g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossCircle));
        g->addData(fit.epochs, fit.finalError[i]);
    }
    ui->panel->replot(QCustomPlot::rpQueuedReplot);
}

void ParameterFitPlotter::xRangeChanged(QCPRange range)
{
    for ( QCPAxisRect *ar : axRects ) {
        QCPAxis *axis = ar->axis(QCPAxis::atBottom);
        axis->blockSignals(true);
        axis->setRange(range);
        axis->blockSignals(false);
    }
    ui->panel->replot();
}

void ParameterFitPlotter::errorRangeChanged(QCPRange range)
{
    for ( QCPAxisRect *ar : axRects ) {
        QCPAxis *axis = ar->axis(QCPAxis::atRight);
        axis->blockSignals(true);
        axis->setRange(QCPRange(0, range.upper));
        axis->blockSignals(false);
    }
    ui->panel->replot();
}

void ParameterFitPlotter::percentileRangeChanged(QCPRange range)
{
    for ( QCPAxisRect *ar : axRects ) {
        QCPAxis *axis = ar->axis(QCPAxis::atLeft, 1);
        axis->blockSignals(true);
        axis->setRange(QCPRange(0, range.upper));
        axis->blockSignals(false);
    }
    ui->panel->replot();
}

//***************************************** summaries *******************************************************

QCPGraph *ParameterFitPlotter::addGraph(QCPAxis *x, QCPAxis *y, const QColor &col,
                                        const QVector<double> &keys, const QVector<double> &values,
                                        const QString &layer, bool visible)
{
    QCPGraph *g = ui->panel->addGraph(x, y);
    g->setPen(QPen(col));
    g->setData(keys, values, true);
    g->setLayer(layer);
    g->setVisible(visible);
    return g;
}

void ParameterFitPlotter::plotSummary()
{
    if ( data.empty() )
        return;

    ui->panel->layer("target")->setVisible(false);

    Filter *filter = nullptr;
    if ( ui->filter->value() > 1 )
        filter = new Filter(FilterMethod::SavitzkyGolayEdge5, 2*int(ui->filter->value()/2) + 1);

    for ( const FitInspector::Group &group : data ) {
        quint32 epochs = 0;
        bool hasFinal = true;
        for ( const FitInspector::Fit &f : group.fits ) {
            epochs = std::max(f.fit().epochs, epochs);
            hasFinal &= f.fit().final;
        }
        QVector<double> keys(epochs);
        for ( size_t i = 0; i < epochs; i++ )
            keys[i] = i;
        for ( size_t i = 0; i < axRects.size(); i++ ) {
            QVector<double> mean(epochs), sem(epochs), median(epochs), max(epochs);
            QVector<double> errMean(epochs), errSEM(epochs), errMedian(epochs), errMax(epochs);
            QVector<double> fMean(1), fSem(1), fMedian(1), fMax(1);
            QVector<double> fErrMean(1), fErrSEM(1), fErrMedian(1), fErrMax(1);
            const AdjustableParam &p = session->project.model().adjustableParams[i];
            if ( p.multiplicative ) {
                getSummary(group.fits, [=](const GAFitter::Output &fit, int ep){
                    // Parameter error relative to target (%) :
                    return 100 * std::fabs(1 - fit.params[ep][i] / fit.targets[i]);
                }, mean, sem, median, max, filter);
                if ( hasFinal )
                    getSummary(group.fits, [=](const GAFitter::Output &fit, int){
                        return 100 * std::fabs(1 - fit.finalParams[i] / fit.targets[i]);
                    }, fMean, fSem, fMedian, fMax);
            } else {
                getSummary(group.fits, [=](const GAFitter::Output &fit, int ep){
                    // Parameter error relative to range (%) :
                    double range = session->gaFitterSettings(fit.resultIndex).constraints[i] == 1
                            ? (session->gaFitterSettings(fit.resultIndex).max[i] - session->gaFitterSettings(fit.resultIndex).min[i])
                            : (p.max - p.min);
                    return 100 * std::fabs((fit.params[ep][i] - fit.targets[i]) / range);
                }, mean, sem, median, max, filter);
                if ( hasFinal )
                    getSummary(group.fits, [=](const GAFitter::Output &fit, int){
                        double range = session->gaFitterSettings(fit.resultIndex).constraints[i] == 1
                                ? (session->gaFitterSettings(fit.resultIndex).max[i] - session->gaFitterSettings(fit.resultIndex).min[i])
                                : (p.max - p.min);
                        return 100 * std::fabs((fit.finalParams[i] - fit.targets[i]) / range);
                    }, fMean, fSem, fMedian, fMax);
            }
            getSummary(group.fits, [=](const GAFitter::Output &fit, int ep) -> double {
                if ( fit.stimSource.type == WaveSource::Deck ) {
                    for ( ; ep >= 0; ep-- )
                        if ( fit.targetStim[ep] == i )
                            return fit.error[ep];
                } else if ( i == 0 ) {
                    return fit.error[ep];
                }
                return 0;
            }, errMean, errSEM, errMedian, errMax, filter);
            if ( hasFinal )
                getSummary(group.fits, [=](const GAFitter::Output &fit, int){
                    return fit.finalError[i];
                }, fErrMean, fErrSEM, fErrMedian, fErrMax);

            QColor col(group.color);
            double opacity = ui->opacity->value()/100.;
            col.setAlphaF(opacity);

            QCPAxis *xAxis = axRects[i]->axis(QCPAxis::atBottom);
            QCPAxis *yAxis2 = axRects[i]->axis(QCPAxis::atRight);
            QCPAxis *yAxis3 = axRects[i]->axis(QCPAxis::atLeft, 1);

            // Mean
            QCPGraph *graph = addGraph(xAxis, yAxis3, col, keys, mean, "mean", ui->param->isChecked());
            if ( hasFinal )
                addGraph(xAxis, yAxis3, col, {double(epochs)}, fMean, "mean", ui->param->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle));

            // Error mean
            QCPGraph *errGraph = addGraph(xAxis, yAxis2, col, keys, errMean, "mean", ui->error->isChecked());
            if ( hasFinal )
                addGraph(xAxis, yAxis2, col, {double(epochs)}, fErrMean, "mean", ui->error->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossCircle));

            // Median
            addGraph(xAxis, yAxis3, col, keys, median, "median", ui->param->isChecked());
            if ( hasFinal )
                addGraph(xAxis, yAxis3, col, {double(epochs)}, fMedian, "median", ui->param->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle));

            // Error median
            addGraph(xAxis, yAxis2, col, keys, errMedian, "median", ui->error->isChecked());
            if ( hasFinal )
                addGraph(xAxis, yAxis2, col, {double(epochs)}, fErrMedian, "median", ui->error->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossCircle));

            // SEM
            col.setAlphaF(0.2*opacity);
            QBrush brush(col);
            col.setAlphaF(0.4*opacity);
            QCPGraph *semGraph = addGraph(xAxis, yAxis3, col, keys, sem, "sem", ui->param->isChecked());
            semGraph->setBrush(brush);
            semGraph->setChannelFillGraph(graph);
            if ( hasFinal )
                addGraph(xAxis, yAxis3, col, {double(epochs)}, fSem, "sem", ui->param->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusSquare));

            // Error SEM
            QCPGraph *errSemGraph = addGraph(xAxis, yAxis2, col, keys, errSEM, "sem", ui->error->isChecked());
            errSemGraph->setBrush(brush);
            errSemGraph->setChannelFillGraph(errGraph);
            if ( hasFinal )
                addGraph(xAxis, yAxis2, col, {double(epochs)}, fErrSEM, "sem", ui->error->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCrossSquare));

            // Max
            col.setAlphaF(opacity);
            addGraph(xAxis, yAxis3, col, keys, max, "max", ui->param->isChecked());
            if ( hasFinal )
                addGraph(xAxis, yAxis3, col, {double(epochs)}, fMax, "max", ui->param->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlus));

            // Error max
            addGraph(xAxis, yAxis2, col, keys, errMax, "max", ui->error->isChecked());
            if ( hasFinal )
                addGraph(xAxis, yAxis2, col, {double(epochs)}, fErrMax, "max", ui->error->isChecked())
                        ->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross));

            if ( i == 0 ) {
                graph->setName(group.label);
                errGraph->setName("");
                graph->addToLegend();
                errGraph->addToLegend();
            }
        }
    }
}

void ParameterFitPlotter::getSummary(std::vector<FitInspector::Fit> fits,
                                     std::function<double(const GAFitter::Output &fit, int epoch)> value,
                                     QVector<double> &mean,
                                     QVector<double> &meanPlusSEM,
                                     QVector<double> &median,
                                     QVector<double> &max,
                                     Filter *filter)
{
    for ( int i = 0; i < mean.size(); i++ ) {
        std::vector<double> values;
        values.reserve(fits.size());
        int n = 0;
        double sum = 0;
        for ( const FitInspector::Fit &f : fits ) {
            if ( i < int(f.fit().epochs) ) {
                double v = value(f.fit(), i);
                values.push_back(v);
                sum += v;
                n++;
            }
        }
        if ( values.empty() )
            break;

        mean[i] = sum / n;

        sum = 0;
        for ( int v : values ) {
            double error = v - mean[i];
            sum += error*error;
        }
        // SEM = (corrected sd)/sqrt(n); corrected sd = sqrt(sum(error^2)/(n-1))
        meanPlusSEM[i] = mean[i] + std::sqrt(sum / (n*(n-1)));

        std::sort(values.begin(), values.end());
        if ( values.size() %2 == 1 )
            median[i] = values[values.size()/2];
        else
            median[i] = (values[values.size()/2] + values[values.size()/2 - 1]) / 2;

        max[i] = values.back();
    }

    if ( filter ) {
        mean = filter->filter(mean);
        meanPlusSEM = filter->filter(meanPlusSEM);
        median = filter->filter(median);
        max = filter->filter(max);
    }
}

void ParameterFitPlotter::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->panel->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), file);
}
