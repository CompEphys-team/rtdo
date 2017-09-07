#include "parameterfitplotter.h"
#include "ui_parameterfitplotter.h"
#include "colorbutton.h"

ParameterFitPlotter::ParameterFitPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ParameterFitPlotter),
    resizing(false),
    summarising(false)
{
    ui->setupUi(this);
    ui->table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(setColumnCount(int)));
    connect(ui->table->verticalHeader(), SIGNAL(sectionResized(int,int,int)), this, SLOT(resizeTableRows(int,int,int)));
    connect(ui->param, &QCheckBox::stateChanged, [=](int state) {
        bool on = state == Qt::Checked;
        for ( QCustomPlot *p : plots ) {
            if ( summarising )
                for ( QCPGraph *g : p->axisRect()->axis(QCPAxis::atLeft, 1)->graphs() )
                    g->setVisible(on);
            else
                for ( QCPGraph *g : p->yAxis->graphs() )
                    g->setVisible(on);
            p->layer("target")->setVisible(on && !summarising);
            p->axisRect()->axis(QCPAxis::atLeft, 1)->setVisible(on && summarising);
            p->yAxis->setVisible(!(on && summarising));
            p->yAxis->setTicks(on);
            p->yAxis->setTickLabels(on);
            p->replot();
        }
    });
    connect(ui->error, &QCheckBox::stateChanged, [=](int state) {
        bool on = state == Qt::Checked;
        for ( QCustomPlot *p : plots ) {
            for ( QCPGraph *g : p->yAxis2->graphs() )
                g->setVisible(on);
            p->yAxis2->setVisible(on);
            p->replot();
        }
    });
    connect(ui->opacity, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), [=](int op){
        double opacity = op/100.;
        for ( QCustomPlot *p : plots ) {
            QList<QCPLayerable*> normalGraphs;
            normalGraphs.append(p->layer("main")->children());
            normalGraphs.append(p->layer("mean")->children());
            normalGraphs.append(p->layer("median")->children());
            normalGraphs.append(p->layer("max")->children());
            for ( QCPLayerable *l : normalGraphs ) {
                QCPGraph *g = qobject_cast<QCPGraph*>(l);
                if ( !g ) continue;
                QPen pen = g->pen();
                QColor col = pen.color();
                col.setAlphaF(opacity);
                pen.setColor(col);
                g->setPen(pen);
            }
            for ( QCPLayerable *l : p->layer("sem")->children() ) {
                QCPGraph *g = qobject_cast<QCPGraph*>(l);
                if ( !g ) continue;
                QColor col = g->pen().color();
                col.setAlphaF(0.4*opacity);
                g->setPen(QPen(col));
                col.setAlphaF(0.2*opacity);
                g->setBrush(QBrush(col));
            }
            p->replot();
        }
    });
    connect(ui->mean, &QCheckBox::toggled, [=](bool on){
        for ( QCustomPlot *p : plots ) {
            p->layer("mean")->setVisible(on);
            p->replot();
        }
    });
    connect(ui->SEM, &QCheckBox::toggled, [=](bool on){
        for ( QCustomPlot *p : plots ) {
            p->layer("sem")->setVisible(on);
            p->replot();
        }
    });
    connect(ui->median, &QCheckBox::toggled, [=](bool on){
        for ( QCustomPlot *p : plots ) {
            p->layer("median")->setVisible(on);
            p->replot();
        }
    });
    connect(ui->max, &QCheckBox::toggled, [=](bool on){
        for ( QCustomPlot *p : plots ) {
            p->layer("max")->setVisible(on);
            p->replot();
        }
    });
    connect(ui->sepcols, &QPushButton::clicked, [=](bool on) {
        for ( int i = 0; i < ui->fits->rowCount(); i++ ) {
            getGraphColorBtn(i)->setColor(on ? QColorDialog::standardColor(i%20) : Qt::blue);
            getErrorColorBtn(i)->setColor(on ? QColorDialog::standardColor(i%20 + 21) : Qt::magenta);
        }
        replot();
    });
    connect(ui->copy, &QPushButton::clicked, [=]() {
        std::vector<int> rows = getSelectedRows(ui->fits);
        clipboard.clear();
        clipboard.reserve(2*rows.size());
        for ( int row : rows ) {
            clipboard.push_back(getGraphColorBtn(row)->color);
            clipboard.push_back(getErrorColorBtn(row)->color);
        }
    });
    connect(ui->paste, &QPushButton::clicked, [=]() {
        std::vector<int> rows = getSelectedRows(ui->fits);
        for ( size_t i = 0; i < rows.size() && 2*i < clipboard.size(); i++ ) {
            getGraphColorBtn(rows[i])->setColor(clipboard[2*i]);
            getErrorColorBtn(rows[i])->setColor(clipboard[2*i+1]);
        }
        replot();
    });

    connect(ui->addGroup, SIGNAL(clicked(bool)), this, SLOT(addGroup()));
    connect(ui->delGroup, SIGNAL(clicked(bool)), this, SLOT(removeGroup()));
}

ParameterFitPlotter::ParameterFitPlotter(Session &session, QWidget *parent) :
    ParameterFitPlotter(parent)
{
    init(&session, false);
}

ParameterFitPlotter::~ParameterFitPlotter()
{
    delete ui;
    for ( QCustomPlot *p : plots )
        delete p;
}

void ParameterFitPlotter::init(Session *session, bool enslave)
{
    this->session = session;
    this->enslaved = enslave;

    // Plots
    for ( QCustomPlot *p : plots )
        delete p;
    plots.resize(session->project.model().adjustableParams.size());
    for ( size_t i = 0; i < plots.size(); i++ ) {
        const AdjustableParam &p = session->project.model().adjustableParams[i];
        QCustomPlot *plot = new QCustomPlot(this);
        plots[i] = plot;
        plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
        plot->xAxis->setLabel("Epoch");
        plot->yAxis->setLabel(QString::fromStdString(p.name));
        QCPAxis *yAxis3 = plot->axisRect()->addAxis(QCPAxis::atLeft);
        plot->addLayer("mean");
        plot->layer("mean")->setVisible(ui->mean->isChecked());
        plot->addLayer("sem");
        plot->layer("sem")->setVisible(ui->SEM->isChecked());
        plot->addLayer("median");
        plot->layer("median")->setVisible(ui->median->isChecked());
        plot->addLayer("max");
        plot->layer("max")->setVisible(ui->max->isChecked());
        plot->addLayer("target");

        plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
        connect(plot->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(xRangeChanged(QCPRange)));
        connect(plot, &QCustomPlot::selectionChangedByUser, [=](){
            QList<QCPAxis *> axes = plot->selectedAxes();
            if ( axes.isEmpty() )
               axes = plot->axisRect()->axes();
            plot->axisRect()->setRangeZoomAxes(axes);
            plot->axisRect()->setRangeDragAxes(axes);
        });
        connect(plot, &QCustomPlot::axisDoubleClick, this, [=](QCPAxis *axis, QCPAxis::SelectablePart, QMouseEvent*) {
            if ( axis == plot->yAxis )
                axis->setRange(p.min, p.max);
            else if ( axis == plot->xAxis )
                axis->rescale();
            else if ( axis == yAxis3 )
                axis->setRange(0, 100);
            else if ( plot->graph(1) ) {
                bool foundRange;
                QCPRange range = plot->graph(1)->getValueRange(foundRange, QCP::sdBoth, plot->xAxis->range());
                if ( foundRange )
                    axis->setRange(range);
            }
            plot->replot();
        });
        plot->axisRect()->setRangeZoomAxes(plot->axisRect()->axes());
        plot->axisRect()->setRangeDragAxes(plot->axisRect()->axes());
        plot->yAxis->setRange(p.min, p.max);
        plot->yAxis->setTicks(ui->param->isChecked());
        plot->yAxis->setTickLabels(ui->param->isChecked());

        plot->yAxis2->setLabel("RMS Error (nA)");
        plot->yAxis2->setVisible(ui->error->isChecked());
        connect(plot->yAxis2, SIGNAL(rangeChanged(QCPRange)), this, SLOT(errorRangeChanged(QCPRange)));

        yAxis3->setLabel(QString("Δ %1 (%%2)").arg(QString::fromStdString(p.name)).arg(p.multiplicative ? "" : " range"));
        yAxis3->setRange(0,100);
        yAxis3->grid()->setVisible(true);
        yAxis3->setVisible(false);
        connect(yAxis3, SIGNAL(rangeChanged(QCPRange)), this, SLOT(percentileRangeChanged(QCPRange)));
    }
    ui->columns->setMaximum(plots.size());
    setColumnCount(ui->columns->value());

    // Enslave to GAFitterWidget
    if ( enslave ) {
        ui->sidebar->setVisible(false);
        connect(&session->gaFitter(), &GAFitter::progress, this, &ParameterFitPlotter::progress);
        connect(&session->gaFitter(), &GAFitter::done, this, [=](){
            const GAFitter::Output &o = session->gaFitter().results().back();
            if ( o.final )
                addFinal(o);
        });
        clear();
    } else {
        for ( int i = 0; i < 2; i++ ) {
            int c = ui->fits->columnCount();
            for ( const AdjustableParam &p : session->project.model().adjustableParams ) {
                ui->fits->insertColumn(c);
                QTableWidgetItem *item = new QTableWidgetItem(QString::fromStdString(p.name));
                item->setToolTip(i ? "Final value" : "Target value");
                ui->fits->setHorizontalHeaderItem(c, item);
                ui->fits->setColumnWidth(c, 70);
                ++c;
            }
            if ( !i ) {
                ui->fits->insertColumn(c);
                ui->fits->setHorizontalHeaderItem(c, new QTableWidgetItem(""));
                ui->fits->setColumnWidth(c, 10);
            }
        }
        ui->fits->setColumnWidth(0, 15);
        ui->fits->setColumnWidth(1, 15);
        ui->fits->setColumnWidth(3, 40);
        ui->fits->setColumnWidth(5, 40);
        connect(&session->gaFitter(), SIGNAL(done()), this, SLOT(updateFits()));
        connect(ui->fits, SIGNAL(itemSelectionChanged()), this, SLOT(replot()));
        connect(ui->groups, SIGNAL(itemSelectionChanged()), this, SLOT(plotSummary()));
        updateFits();
    }
    plots[0]->xAxis->setRange(0, 1000);
}

void ParameterFitPlotter::setColumnCount(int n)
{
    ui->table->clear();
    ui->table->setRowCount(std::ceil(double(plots.size())/n));
    ui->table->setColumnCount(n);
    size_t i = 0;
    for ( int row = 0; row < ui->table->rowCount(); row++ ) {
        for ( int col = 0; col < ui->table->columnCount(); col++ ) {
            QWidget *widget = new QWidget();
            QGridLayout *layout = new QGridLayout(widget);
            layout->addWidget(plots[i]);
            layout->setMargin(0);
            widget->setLayout(layout);
            ui->table->setCellWidget(row, col, widget);
            if ( ++i >= plots.size() )
                row = col = plots.size();
        }
    }
}

void ParameterFitPlotter::resizeTableRows(int, int, int size)
{
    if ( resizing )
        return;
    resizing = true;
    for ( int i = 0; i < ui->table->rowCount(); i++ )
        ui->table->verticalHeader()->resizeSection(i, size);
    resizing = false;
}

void ParameterFitPlotter::updateFits()
{
    for ( size_t i = ui->fits->rowCount(); i < session->gaFitter().results().size(); i++ ) {
        const GAFitter::Output &fit = session->gaFitter().results().at(i);
        ui->fits->insertRow(i);
        ui->fits->setVerticalHeaderItem(i, new QTableWidgetItem(QString::number(i)));
        ColorButton *c = new ColorButton();
        c->setColor(ui->sepcols->isChecked() ? QColorDialog::standardColor(i%20) : Qt::blue);
        ui->fits->setCellWidget(i, 0, c);
        c = new ColorButton();
        c->setColor(ui->sepcols->isChecked() ? QColorDialog::standardColor(i%20 + 21) : Qt::magenta);
        ui->fits->setCellWidget(i, 1, c);
        ui->fits->setItem(i, 2, new QTableWidgetItem(QString::number(fit.deck.idx)));
        ui->fits->setItem(i, 3, new QTableWidgetItem(QString::number(fit.epochs)));
        ui->fits->setItem(i, 4, new QTableWidgetItem(QString::number(fit.settings.randomOrder)));
        ui->fits->setItem(i, 5, new QTableWidgetItem(QString::number(fit.settings.crossover, 'g', 2)));
        ui->fits->setItem(i, 6, new QTableWidgetItem(fit.settings.decaySigma ? "Y" : "N"));
        ui->fits->setItem(i, 7, new QTableWidgetItem(QString::number(fit.settings.targetType)));
        for ( size_t j = 0; j < session->project.model().adjustableParams.size(); j++ )
            ui->fits->setItem(i, 8+j, new QTableWidgetItem(QString::number(fit.targets[j], 'g', 3)));
        if ( fit.final )
            for ( size_t j = 0, np = session->project.model().adjustableParams.size(); j < np; j++ )
                ui->fits->setItem(i, 8+np+1+j, new QTableWidgetItem(QString::number(fit.finalParams[j], 'g', 3)));
    }
}

void ParameterFitPlotter::replot()
{
    bool initial = true;
    std::vector<int> rows = getSelectedRows(ui->fits);
    if ( rows.empty() )
        return;

    for ( QCustomPlot *p : plots ) {
        p->layer("target")->setVisible(ui->param->isChecked());
        p->yAxis->setVisible(true);
        p->axisRect()->axis(QCPAxis::atLeft, 1)->setVisible(false);
        p->clearItems();
        p->clearGraphs();
    }
    summarising = false;

    for ( int row : rows ) {
        const GAFitter::Output fit = session->gaFitter().results().at(row);
        QVector<double> keys(fit.epochs);
        for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ )
            keys[epoch] = epoch;
        for ( size_t i = 0; i < plots.size(); i++ ) {
            QCPItemStraightLine *line = new QCPItemStraightLine(plots[i]);
            line->setLayer("target");
            line->setPen(QPen(getGraphColorBtn(row)->color));
            line->point1->setCoords(0, fit.targets[i]);
            line->point2->setCoords(1, fit.targets[i]);

            QVector<double> values(fit.epochs), errors, errKey;
            errors.reserve(fit.epochs);
            errKey.reserve(fit.epochs);
            for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ ) {
                values[epoch] = fit.params[epoch][i];
                if ( fit.stimIdx[epoch] == i ) {
                    errors.push_back(fit.error[epoch]);
                    errKey.push_back(epoch);
                }
            }

            QCPGraph *graph = plots[i]->addGraph();
            QColor col(getGraphColorBtn(row)->color);
            col.setAlphaF(ui->opacity->value()/100.);
            graph->setPen(QPen(col));
            graph->setData(keys, values, true);
            graph->setVisible(ui->param->isChecked());
            if ( ui->rescale->isChecked() ) {
                plots[i]->xAxis->rescale();
                plots[i]->yAxis->rescale();
            }

            QCPGraph *errGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
            QColor errCol(getErrorColorBtn(row)->color);
            errCol.setAlphaF(ui->opacity->value()/100.);
            errGraph->setPen(QPen(errCol));
            errGraph->setLineStyle(QCPGraph::lsStepLeft);
            errGraph->setData(errKey, errors, true);
            errGraph->setVisible(ui->error->isChecked());
            if ( ui->rescale->isChecked() ) {
                if ( initial ) {
                    plots[i]->yAxis2->setRange(0,1); // Reset range when selecting a new fit
                    initial = false;
                }
                bool found;
                QCPRange range = errGraph->getValueRange(found);
                if ( found && range.upper > plots[i]->yAxis2->range().upper )
                    plots[i]->yAxis2->setRangeUpper(range.upper);
            }
        }

        if ( fit.final )
            addFinal(fit);
    }

    for ( QCustomPlot *p : plots )
        p->replot();
}

void ParameterFitPlotter::clear()
{
    for ( QCustomPlot *plot : plots ) {
        plot->clearGraphs();
        plot->clearItems();
        plot->addGraph()->setVisible(ui->param->isChecked());
        plot->addGraph(0, plot->yAxis2)->setVisible(ui->error->isChecked());
        plot->graph(1)->setLineStyle(QCPGraph::lsStepLeft);
        plot->graph(1)->setPen(QPen(Qt::magenta));
        plot->xAxis->moveRange(-plot->xAxis->range().lower);
        plot->replot();
    }
}

std::vector<int> ParameterFitPlotter::getSelectedRows(QTableWidget* table)
{
    QList<QTableWidgetSelectionRange> selection = table->selectedRanges();
    std::vector<int> rows;
    for ( auto range : selection )
        for ( int i = range.topRow(); i <= range.bottomRow(); i++)
            rows.push_back(i);
    return rows;
}

ColorButton *ParameterFitPlotter::getGraphColorBtn(int row)
{
    return qobject_cast<ColorButton*>(ui->fits->cellWidget(row, 0));
}

ColorButton *ParameterFitPlotter::getErrorColorBtn(int row)
{
    return qobject_cast<ColorButton*>(ui->fits->cellWidget(row, 1));
}

ColorButton *ParameterFitPlotter::getGroupColorBtn(int row)
{
    return qobject_cast<ColorButton*>(ui->groups->cellWidget(row, 0));
}

void ParameterFitPlotter::progress(quint32 epoch)
{
    GAFitter::Output fit = session->gaFitter().currentResults();
    if ( epoch == 0 ) {
        for ( size_t i = 0; i < plots.size(); i++ ) {
            QCPItemStraightLine *line = new QCPItemStraightLine(plots[i]);
            line->setLayer("target");
            line->point1->setCoords(0, fit.targets[i]);
            line->point2->setCoords(1, fit.targets[i]);
        }
    }
    for ( size_t i = 0; i < plots.size(); i++ ) {
        plots[i]->graph(0)->addData(epoch, fit.params[epoch][i]);
    }
    plots[fit.stimIdx[epoch]]->graph(1)->addData(epoch, fit.error[epoch]);

    double xUpper = plots[0]->xAxis->range().upper;
    if ( double(epoch-1) <= xUpper && double(epoch) > xUpper) {
        for ( QCustomPlot *plot : plots ) {
            plot->xAxis->blockSignals(true);
            plot->xAxis->moveRange(1);
            plot->xAxis->blockSignals(false);
        }
    }

    for ( QCustomPlot *p : plots )
        p->replot(QCustomPlot::rpQueuedReplot);
}

void ParameterFitPlotter::addFinal(const GAFitter::Output &fit)
{
    for ( size_t i = 0; i < plots.size(); i++ ) {
        const QCPGraph *gParam = plots[i]->graph(plots[i]->graphCount()-2), *gErr = plots[i]->graph();

        QCPGraph *g = plots[i]->addGraph(gParam->keyAxis(), gParam->valueAxis());
        g->setVisible(gParam->visible());
        g->setPen(gParam->pen());
        g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle));
        g->addData(fit.epochs, fit.finalParams[i]);

        g = plots[i]->addGraph(gErr->keyAxis(), gErr->valueAxis());
        g->setVisible(gErr->visible());
        g->setPen(gErr->pen());
        g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssPlusCircle));
        g->addData(fit.epochs, fit.finalError[i]);

        plots[i]->replot(QCustomPlot::rpQueuedReplot);
    }
}

void ParameterFitPlotter::xRangeChanged(QCPRange range)
{
    for ( QCustomPlot *plot : plots ) {
        plot->xAxis->blockSignals(true);
        plot->xAxis->setRange(range);
        plot->replot();
        plot->xAxis->blockSignals(false);
    }
}

void ParameterFitPlotter::errorRangeChanged(QCPRange range)
{
    for ( QCustomPlot *plot : plots ) {
        plot->yAxis2->blockSignals(true);
        plot->yAxis2->setRange(QCPRange(0, range.upper));
        plot->replot();
        plot->yAxis2->blockSignals(false);
    }
}

void ParameterFitPlotter::percentileRangeChanged(QCPRange range)
{
    for ( QCustomPlot *plot : plots ) {
        QCPAxis *y3 = plot->axisRect()->axis(QCPAxis::atLeft, 1);
        y3->blockSignals(true);
        y3->setRange(QCPRange(0, range.upper));
        plot->replot();
        y3->blockSignals(false);
    }
}

//***************************************** summaries *******************************************************

void ParameterFitPlotter::addGroup()
{
    std::vector<int> group = getSelectedRows(ui->fits);
    if ( group.empty() )
        return;
    std::sort(group.begin(), group.end());
    groups.push_back(group);

    int row = ui->groups->rowCount();
    ui->groups->insertRow(row);
    ColorButton *c = new ColorButton();
    c->setColor(QColorDialog::standardColor(row % 20));
    ui->groups->setCellWidget(row, 0, c);
    connect(c, SIGNAL(colorChanged(QColor)), this, SLOT(plotSummary()));

    QString label = "Fits ";
    for ( int i = 0, last = group.size()-1; i <= last; i++ ) {
        int beginning = group[i];
        while ( i < last && group[i+1] == group[i] + 1 )
            ++i;
        if ( group[i] > beginning )
            label.append(QString("%1-%2").arg(beginning).arg(group[i]));
        else
            label.append(QString::number(group[i]));
        if ( i < last )
            label.append("; ");
    }
    QTableWidgetItem *item = new QTableWidgetItem(label);
    ui->groups->setItem(row, 1, item);
}

void ParameterFitPlotter::removeGroup()
{
    std::vector<int> rows = getSelectedRows(ui->groups);
    std::sort(rows.begin(), rows.end(), [](int a, int b){return a > b;}); // descending
    for ( int row : rows ) {
        ui->groups->removeRow(row);
        groups.erase(groups.begin() + row);
    }
}

void ParameterFitPlotter::plotSummary()
{
    std::vector<int> rows = getSelectedRows(ui->groups);
    if ( rows.empty() )
        return;

    for ( QCustomPlot *p : plots ) {
        p->clearGraphs();
        p->layer("target")->setVisible(false);
        p->yAxis->setVisible(false);
        p->axisRect()->axis(QCPAxis::atLeft, 1)->setVisible(true);
    }
    summarising = true;

    for ( int row : rows ) {
        quint32 epochs = 0;
        for ( int fit : groups[row] )
            epochs = std::max(session->gaFitter().results().at(fit).epochs, epochs);
        QVector<double> keys(epochs);
        for ( size_t i = 0; i < epochs; i++ )
            keys[i] = i;
        for ( size_t i = 0; i < plots.size(); i++ ) {
            QVector<double> mean(epochs), sem(epochs), median(epochs), max(epochs);
            QVector<double> errMean(epochs), errSEM(epochs), errMedian(epochs), errMax(epochs);
            const AdjustableParam &p = session->project.model().adjustableParams[i];
            if ( p.multiplicative ) {
                getSummary(groups[row], [=](const GAFitter::Output &fit, int ep){
                    return 100 * std::fabs(1 - fit.params[ep][i] / fit.targets[i]); // Parameter error relative to target (%)
                }, mean, sem, median, max);
            } else {
                getSummary(groups[row], [=](const GAFitter::Output &fit, int ep){
                    return 100 * std::fabs((fit.params[ep][i] - fit.targets[i]) / (p.max - p.min)); // Parameter error relative to range (%)
                }, mean, sem, median, max);
            }
            getSummary(groups[row], [=](const GAFitter::Output &fit, int ep) -> double {
                for ( ; ep >= 0; ep-- )
                    if ( fit.stimIdx[ep] == i )
                        return fit.error[ep];
                return 0;
            }, errMean, errSEM, errMedian, errMax);
            QColor col(getGroupColorBtn(row)->color);
            double opacity = ui->opacity->value()/100.;
            col.setAlphaF(opacity);

            QCPAxis *yAxis3 = plots[i]->axisRect()->axis(QCPAxis::atLeft, 1);

            // Mean
            QCPGraph *graph = plots[i]->addGraph(0, yAxis3);
            graph->setPen(QPen(col));
            graph->setData(keys, mean, true);
            graph->setLayer("mean");
            graph->setVisible(ui->param->isChecked());

            // Error mean
            QCPGraph *errGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
            errGraph->setPen(QPen(col));
            errGraph->setData(keys, errMean, true);
            errGraph->setLayer("mean");
            errGraph->setVisible(ui->error->isChecked());

            // Median
            QCPGraph *medianGraph = plots[i]->addGraph(0, yAxis3);
            medianGraph->setPen(QPen(col));
            medianGraph->setData(keys, median, true);
            medianGraph->setLayer("median");
            medianGraph->setVisible(ui->param->isChecked());

            // Error median
            QCPGraph *errMedianGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
            errMedianGraph->setPen(QPen(col));
            errMedianGraph->setData(keys, errMedian, true);
            errMedianGraph->setLayer("median");
            errMedianGraph->setVisible(ui->error->isChecked());

            // SEM
            QCPGraph *semGraph = plots[i]->addGraph(0, yAxis3);
            col.setAlphaF(0.2*opacity);
            QBrush brush(col);
            col.setAlphaF(0.4*opacity);
            semGraph->setPen(QPen(col));
            semGraph->setBrush(brush);
            semGraph->setChannelFillGraph(graph);
            semGraph->setData(keys, sem, true);
            semGraph->setLayer("sem");
            semGraph->setVisible(ui->param->isChecked());

            // Error SEM
            QCPGraph *errSemGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
            errSemGraph->setPen(QPen(col));
            errSemGraph->setBrush(brush);
            errSemGraph->setChannelFillGraph(errGraph);
            errSemGraph->setData(keys, errSEM, true);
            errSemGraph->setLayer("sem");
            errSemGraph->setVisible(ui->error->isChecked());

            // Max
            QCPGraph *maxGraph = plots[i]->addGraph(0, yAxis3);
            col.setAlphaF(0.6*opacity);
            QPen pen(col);
            pen.setStyle(Qt::DotLine);
            maxGraph->setPen(pen);
            maxGraph->setData(keys, max, true);
            maxGraph->setLayer("max");
            maxGraph->setVisible(ui->param->isChecked());

            // Error max
            QCPGraph *errMaxGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
            errMaxGraph->setPen(pen);
            errMaxGraph->setData(keys, errMax, true);
            errMaxGraph->setLayer("max");
            errMaxGraph->setVisible(ui->error->isChecked());
        }
    }

    for ( QCustomPlot *p : plots )
        p->replot();
}

void ParameterFitPlotter::getSummary(std::vector<int> fits,
                                     std::function<double(const GAFitter::Output &fit, int epoch)> value,
                                     QVector<double> &mean,
                                     QVector<double> &meanPlusSEM,
                                     QVector<double> &median,
                                     QVector<double> &max)
{
    for ( int i = 0; i < mean.size(); i++ ) {
        std::vector<double> values;
        values.reserve(fits.size());
        int n = 0;
        double sum = 0;
        for ( int row : fits ) {
            const GAFitter::Output &fit = session->gaFitter().results().at(row);
            if ( i < (int)fit.epochs ) {
                double v = value(fit, i);
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
}
