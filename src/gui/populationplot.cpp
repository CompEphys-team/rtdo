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


#include "populationplot.h"
#include "ui_populationplot.h"
#include "populationsaver.h"

PopulationPlot::PopulationPlot(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PopulationPlot)
{
    ui->setupUi(this);
    setAttribute(Qt::WA_DeleteOnClose);
    connect(ui->slider, SIGNAL(valueChanged(int)), this, SLOT(resizePanel()));
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(clearPlotLayout()));
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(buildPlotLayout()));

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
        const GAFitterSettings &settings = enslaved ? session->gaFitterSettings() : session->gaFitterSettings(session->gaFitter().results()[ui->fits->currentIndex()].resultIndex);
        if ( axis == ar->axis(QCPAxis::atLeft, 0) ) {
            for ( size_t i = 0; i < axRects.size(); i++ ) {
                if ( axRects[i] == ar ) {
                    axis->setRange(settings.min[i], settings.max[i]);
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

    connect(ui->fits, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &PopulationPlot::replot);
    connect(ui->bins, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &PopulationPlot::replot);
    connect(ui->mode, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &PopulationPlot::replot);
    connect(ui->bestCost, &QCheckBox::toggled, this, &PopulationPlot::replot);
    connect(ui->bestVali, &QCheckBox::toggled, this, &PopulationPlot::replot);
    connect(ui->target, &QCheckBox::toggled, this, &PopulationPlot::replot);
}

PopulationPlot::PopulationPlot(Session &session, QWidget *parent) :
    PopulationPlot(parent)
{
    init(&session, false);
}

void PopulationPlot::init(Session *session, bool enslave)
{
    this->session = session;
    this->enslaved = enslave;

    // Plots
    axRects.resize(session->project.model().adjustableParams.size());
    for ( size_t i = 0; i < axRects.size(); i++ ) {
        const AdjustableParam &p = session->project.model().adjustableParams[i];
        QCPAxisRect *ar = new QCPAxisRect(ui->panel);
        axRects[i] = ar;

        QCPAxis *xAxis = ar->axis(QCPAxis::atBottom);
        xAxis->setLabel("Epoch");
        xAxis->setRange(0, 1000);
        connect(xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(xRangeChanged(QCPRange)));

        QCPAxis *yAxis = ar->axis(QCPAxis::atLeft, 0);
        yAxis->setLabel(QString::fromStdString(p.name));
        yAxis->setRange(p.min, p.max);

        ar->setRangeDragAxes(ar->axes());
        ar->setRangeZoomAxes(ar->axes());
    }
    ui->panel->legend->setVisible(true);
    ui->panel->setAutoAddPlottableToLegend(false);
    ui->panel->legend->setFillOrder(QCPLegend::foColumnsFirst);
    ui->panel->legend->setWrap(2);
    ui->panel->axisRect()->insetLayout()->take(ui->panel->legend);
    ui->columns->setMaximum(axRects.size());

    // Enslave to GAFitterWidget
//    if ( enslave ) {
//        ui->fits->setVisible(false);
//        connect(&session->gaFitter(), &GAFitter::starting, this, &PopulationPlot::clear);
//        connect(&session->gaFitter(), &GAFitter::progress, this, &PopulationPlot::progress);
//        connect(&session->gaFitter(), &GAFitter::done, this, [=](){
//            const GAFitter::Output &o = session->gaFitter().results().back();
//            if ( o.final )
//                addFinal(o);
//        });
//        clear();
//    } else {

    updateCombos();
    connect(&session->gaFitter(), &GAFitter::done, this, &PopulationPlot::updateCombos);

    replot();
    QTimer::singleShot(10, this, &PopulationPlot::resizePanel);
}

PopulationPlot::~PopulationPlot()
{
    delete ui;
    delete lib;
}

void PopulationPlot::updateCombos()
{
    for ( size_t i = ui->fits->count(); i < session->gaFitter().results().size(); i++ )
        ui->fits->addItem(QString("Fit %1 (%2)").arg(i).arg(session->gaFitter().results().at(i).resultIndex, 4, 10, QChar('0')));

    for ( size_t i = ui->validations->count(); i < session->gaFitter().validations().size(); i++ ) {
        const GAFitter::Validation &val = session->gaFitter().validations().at(i);
        ui->validations->addItem(QString("Validation %1 for fit %2").arg(val.resultIndex, 4, 10, QChar('0')).arg(val.fitIdx));
    }
}

void PopulationPlot::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    resizePanel();
}

void PopulationPlot::resizePanel()
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

void PopulationPlot::clearPlotLayout()
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

void PopulationPlot::buildPlotLayout()
{
    ui->panel->plotLayout()->clear();

    QCPLayoutGrid *graphLayout = new QCPLayoutGrid();
    ui->panel->plotLayout()->addElement(0, 0, graphLayout);

    size_t i = 0;
    int n = ui->columns->value();
    for ( int row = 0; row < std::ceil(double(axRects.size())/n); row++ ) {
        for ( int col = 0; col < n; col++ ) {
            graphLayout->addElement(row, col, axRects[i]);
            if ( ++i >= axRects.size() )
                row = col = axRects.size(); // break
        }
    }
    ui->panel->replot(QCustomPlot::rpQueuedRefresh);
    resizePanel();
}

void PopulationPlot::replot()
{
    ui->panel->clearPlottables();
    clearPlotLayout();
    ui->panel->plotLayout()->addElement(new QCPAxisRect(ui->panel)); // Prevent debug output from within QCP on adding items

    if ( enslaved || ui->fits->currentIndex() < 0 ) {
        buildPlotLayout();
        return;
    }

    const int fitIdx = ui->fits->currentIndex();
    const int valIdx = ui->validations->currentIndex();
    const int nBins = ui->bins->value();
    int mode = ui->mode->currentIndex();
    const GAFitter::Output &fit = session->gaFitter().results().at(fitIdx);
    const GAFitterSettings &settings = session->gaFitterSettings(fit.resultIndex);
    const GAFitter::Validation *validation = nullptr;

    if ( lib == nullptr )
        lib = new UniversalLibrary(session->project, false, true);
    QFile basefile(session->resultFilePath(session->gaFitter().results().at(fitIdx).resultIndex));
    PopLoader loader(basefile, *lib);

    if ( valIdx < 0 || session->gaFitter().validations().at(valIdx).fitIdx != fitIdx ) {
        if ( mode == 2 || mode == 5 || mode == 6 )
            mode = -1;
    } else {
        validation =& session->gaFitter().validations().at(valIdx);
    }

    std::vector<QCPColorMap*> maps(axRects.size(), nullptr);
    std::vector<QCPGraph*> bestCostG(axRects.size(), nullptr);
    std::vector<QCPGraph*> bestValiG(axRects.size(), nullptr);
    for ( size_t i = 0; i < axRects.size(); i++ ) {
        maps[i] = new QCPColorMap(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft));
        maps[i]->data()->setSize(fit.epochs, nBins);
        maps[i]->data()->setRange(QCPRange {0, double(fit.epochs)}, QCPRange {settings.min[i], settings.max[i]});

        if ( validation ) {
            bestValiG[i] = new QCPGraph(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft));
            bestValiG[i]->setVisible(ui->bestVali->isChecked());
            bestValiG[i]->setPen(QPen(Qt::green));
            bestValiG[i]->setLineStyle(QCPGraph::lsNone);
            bestValiG[i]->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 4));
        }
        bestCostG[i] = new QCPGraph(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft));
        bestCostG[i]->setVisible(ui->bestCost->isChecked());
        bestCostG[i]->setLineStyle(QCPGraph::lsNone);
        bestCostG[i]->setScatterStyle(QCPScatterStyle::ssCross);
    }


    for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ ) {
        loader.load(epoch, *lib);
        for ( size_t i = 0; i < axRects.size(); i++ ) {
            std::vector<double> hist(nBins, 0);
            std::vector<std::vector<double>> binnedValues(nBins);
            double histTotal = mode ? 0 : 1;
            double bestCost = lib->summary[0];
            double bestVali = (validation && !validation->error[epoch].empty()) ? validation->error[epoch][0] : -1;
            int bestCostIdx = 0, bestValiIdx = 0;
            for ( size_t j = 0; j < lib->NMODELS; j++ ) {
                const QCPRange &range = maps[i]->data()->valueRange();
                int bucket = std::min(int((lib->adjustableParams[i][j] - range.lower) / range.size() * nBins), nBins-1);
                if ( mode == 0 ) // unweighted
                    ++hist[bucket];
                else if ( mode == 1 ) { // weighted by fit cost
                    double invCost = 1/lib->summary[j];
                    hist[bucket] += invCost;
                    histTotal += invCost;
                } else if ( mode == 2 && !validation->error[epoch].empty() ) { // weighted by validation
                    double invError = 1/validation->error[epoch][j];
                    hist[bucket] += invError;
                    histTotal += invError;
                } else if ( mode == 3 ) {
                    binnedValues[bucket].push_back(lib->summary[j]);
                } else if ( mode == 4 ) {
                    if ( binnedValues[bucket].empty() )
                        binnedValues[bucket].push_back(lib->summary[j]);
                    else
                        binnedValues[bucket][0] = std::min(binnedValues[bucket][0], double(lib->summary[j]));
                } else if ( mode == 5 && !validation->error[epoch].empty() ) {
                    binnedValues[bucket].push_back(validation->error[epoch][j]);
                } else if ( mode == 6 && !validation->error[epoch].empty() ) {
                    if ( binnedValues[bucket].empty() )
                        binnedValues[bucket].push_back(validation->error[epoch][j]);
                    else
                        binnedValues[bucket][0] = std::min(binnedValues[bucket][0], validation->error[epoch][j]);
                }

                if ( lib->summary[j] < bestCost ) {
                    bestCost = lib->summary[j];
                    bestCostIdx = j;
                }
                if ( validation && !validation->error[epoch].empty() && validation->error[epoch][j] < bestVali ) {
                    bestVali = validation->error[epoch][j];
                    bestValiIdx = j;
                }
            }

            if ( mode < 3 ) {
                for ( int b = 0; b < nBins; b++ )
                    maps[i]->data()->setCell(epoch, b, hist[b] / histTotal);
            } else if ( mode == 3 || mode == 5 ) {
                for ( int b = 0; b < nBins; b++ ) {
                    double val = 0;
                    for ( const double &v : binnedValues[b] )
                        val += v;
                    maps[i]->data()->setCell(epoch, b, (val > 0) ? binnedValues[b].size() / val : 0); // inverse mean
                }
            } else if ( mode == 4 || mode == 6 ) {
                for ( int b = 0; b < nBins; b++ )
                    maps[i]->data()->setCell(epoch, b, binnedValues[b].empty() ? 0 : 1/binnedValues[b][0]); // inverse min
            }

            for ( size_t paramIdx = 0; paramIdx < bestCostG.size(); paramIdx++ ) {
                bestCostG[paramIdx]->addData(epoch, lib->adjustableParams[paramIdx][bestCostIdx]);
                if ( validation && !validation->error[epoch].empty() )
                    bestValiG[paramIdx]->addData(epoch, lib->adjustableParams[paramIdx][bestValiIdx]);
            }
        }
    }

    if ( ui->target->isChecked() ) {
        for ( size_t i = 0; i < axRects.size(); i++ ) {
            QCPItemStraightLine *line = new QCPItemStraightLine(ui->panel);
            line->setPen(QPen(Qt::magenta));
            line->point1->setAxes(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft));
            line->point2->setAxes(axRects[i]->axis(QCPAxis::atBottom), axRects[i]->axis(QCPAxis::atLeft));
            line->point1->setCoords(0, fit.targets[i]);
            line->point2->setCoords(1, fit.targets[i]);
            line->setClipAxisRect(axRects[i]);
            line->setClipToAxisRect(true);
        }
    }

    QCPColorGradient gradient(QCPColorGradient::gpHot);
    gradient.setColorStopAt(0, Qt::darkGray);
    gradient.setColorStopAt(__DBL_MIN__, Qt::black);

    for ( size_t i = 0; i < axRects.size(); i++ ) {
        maps[i]->rescaleDataRange();
        maps[i]->setGradient(gradient);
    }

    buildPlotLayout();

    ui->panel->rescaleAxes();
}

void PopulationPlot::xRangeChanged(QCPRange range)
{
    for ( QCPAxisRect *ar : axRects ) {
        QCPAxis *axis = ar->axis(QCPAxis::atBottom);
        axis->blockSignals(true);
        axis->setRange(range);
        axis->blockSignals(false);
    }
    ui->panel->replot();
}

void PopulationPlot::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->panel->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(),
                      QString("Fit %1").arg(session->gaFitter().results().at(ui->fits->currentIndex()).resultIndex));
}
