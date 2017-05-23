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
            for ( QCPGraph *g : p->yAxis->graphs() )
                g->setVisible(on);
            p->item(0)->setVisible(on && summarising); // This should be the target value line
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
    connect(ui->opacity, SIGNAL(valueChanged(int)), this, SLOT(replot()));
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
        QCPItemStraightLine *line = new QCPItemStraightLine(plot);
        line->point1->setCoords(0, p.initial);
        line->point2->setCoords(1, p.initial);
        line->setVisible(ui->param->isChecked());

        plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
        connect(plot->xAxis, SIGNAL(rangeChanged(QCPRange)), this, SLOT(rangeChanged(QCPRange)));
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

        plot->yAxis2->setLabel("Error");
        plot->yAxis2->setVisible(ui->error->isChecked());
        connect(plot->yAxis2, SIGNAL(rangeChanged(QCPRange)), this, SLOT(errorRangeChanged(QCPRange)));
    }
    ui->columns->setMaximum(plots.size());
    setColumnCount(ui->columns->value());

    // Enslave to GAFitterWidget
    if ( enslave ) {
        ui->sidebar->setVisible(false);
        connect(&session->gaFitter(), &GAFitter::progress, this, &ParameterFitPlotter::progress);
        clear();
    } else {
        connect(&session->gaFitter(), SIGNAL(done()), this, SLOT(updateFits()));
        connect(ui->fits, SIGNAL(itemSelectionChanged()), this, SLOT(replot()));
        connect(ui->groups, SIGNAL(itemSelectionChanged()), this, SLOT(plotSummary()));
        connect(ui->mean, SIGNAL(toggled(bool)), this, SLOT(plotSummary()));
        connect(ui->SEM, SIGNAL(toggled(bool)), this, SLOT(plotSummary()));
        connect(ui->max, SIGNAL(toggled(bool)), this, SLOT(plotSummary()));
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
        ui->fits->setItem(i, 2, new QTableWidgetItem(QString("Fit %1 (%2 epochs, %3)").arg(i).arg(fit.epochs).arg(fit.deck.prettyName())));
        ColorButton *c = new ColorButton();
        c->setColor(ui->sepcols->isChecked() ? QColorDialog::standardColor(i%20) : Qt::blue);
        ui->fits->setCellWidget(i, 0, c);
        c = new ColorButton();
        c->setColor(ui->sepcols->isChecked() ? QColorDialog::standardColor(i%20 + 21) : Qt::magenta);
        ui->fits->setCellWidget(i, 1, c);
    }
}

void ParameterFitPlotter::replot()
{
    bool initial = true;
    std::vector<int> rows = getSelectedRows(ui->fits);
    if ( rows.empty() )
        return;

    for ( QCustomPlot *p : plots ) {
        p->item(0)->setVisible(ui->param->isChecked());
        p->clearGraphs();
    }

    for ( int row : rows ) {
        const GAFitter::Output fit = session->gaFitter().results().at(row);
        QVector<double> keys(fit.epochs);
        for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ )
            keys[epoch] = epoch;
        for ( size_t i = 0; i < plots.size(); i++ ) {
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
    }

    for ( QCustomPlot *p : plots )
        p->replot();
}

void ParameterFitPlotter::clear()
{
    for ( QCustomPlot *plot : plots ) {
        plot->clearGraphs();
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

void ParameterFitPlotter::rangeChanged(QCPRange range)
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
        p->item(0)->setVisible(false);
    }

    for ( int row : rows ) {
        quint32 epochs = 0;
        for ( int fit : groups[row] )
            epochs = std::max(session->gaFitter().results().at(fit).epochs, epochs);
        QVector<double> keys(epochs);
        for ( size_t i = 0; i < epochs; i++ )
            keys[i] = i;
        for ( size_t i = 0; i < plots.size(); i++ ) {
            QVector<double> mean(epochs, 0), sem(epochs, 0), max(epochs, 0);
            QVector<double> errMean(epochs, 0), errSEM(epochs, 0), errMax(epochs, 0);
            getSummary(groups[row], [=](const GAFitter::Output &fit, int ep){
                return std::fabs(fit.params[ep][i] - session->project.model().adjustableParams[i].initial);
            }, mean, sem, max);
            getSummary(groups[row], [=](const GAFitter::Output &fit, int ep) -> double {
                for ( ; ep >= 0; ep-- )
                    if ( fit.stimIdx[ep] == i )
                        return fit.error[ep];
                return 0;
            }, errMean, errSEM, errMax);
            QColor col(getGroupColorBtn(row)->color);
            double opacity = ui->opacity->value()/100.;
            col.setAlphaF(opacity);

            if ( ui->mean->isChecked() ) {
                QCPGraph *graph = plots[i]->addGraph();
                graph->setPen(QPen(col));
                graph->setData(keys, mean, true);

                QCPGraph *errGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
                errGraph->setPen(QPen(col));
                errGraph->setData(keys, errMean, true);

                if ( ui->SEM->isChecked() ) {
                    QCPGraph *semGraph = plots[i]->addGraph();
                    col.setAlphaF(0.2*opacity);
                    QBrush brush(col);
                    col.setAlphaF(0.4*opacity);
                    semGraph->setPen(QPen(col));
                    semGraph->setBrush(brush);
                    semGraph->setChannelFillGraph(graph);
                    semGraph->setData(keys, sem, true);

                    QCPGraph *errSemGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
                    errSemGraph->setPen(QPen(col));
                    errSemGraph->setBrush(brush);
                    errSemGraph->setChannelFillGraph(errGraph);
                    errSemGraph->setData(keys, errSEM, true);
                }
            }

            if ( ui->max->isChecked() ) {
                QCPGraph *maxGraph = plots[i]->addGraph();
                col.setAlphaF(0.6*opacity);
                QPen pen(col);
                pen.setStyle(Qt::DotLine);
                maxGraph->setPen(pen);
                maxGraph->setData(keys, max, true);

                QCPGraph *errMaxGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
                errMaxGraph->setPen(pen);
                errMaxGraph->setData(keys, errMax, true);
            }
        }
    }

    for ( QCustomPlot *p : plots )
        p->replot();
}

void ParameterFitPlotter::getSummary(std::vector<int> fits,
                                     std::function<double(const GAFitter::Output &fit, int epoch)> value,
                                     QVector<double> &mean,
                                     QVector<double> &meanPlusSEM,
                                     QVector<double> &max)
{
    std::vector<int> n(mean.size(), 0);
    for ( int row : fits ) {
        const GAFitter::Output &fit = session->gaFitter().results().at(row);
        for ( size_t i = 0; i < fit.epochs; i++ ) {
            double diff = value(fit, i);
            mean[i] += diff;
            n[i]++;
            if ( diff > max[i] )
                max[i] = diff;
        }
    }
    for ( int i = 0; i < mean.size(); i++ )
        mean[i] /= n[i];
    for ( int row : fits ) {
        const GAFitter::Output fit = session->gaFitter().results().at(row);
        for ( size_t i = 0; i < fit.epochs; i++ ) {
            double diff = value(fit, i);
            double error = diff - mean[i];
            meanPlusSEM[i] += error*error;
        }
    }
    for ( int i = 0; i < meanPlusSEM.size(); i++ )
        // SEM = (corrected sd)/sqrt(n); corrected sd = sqrt(sum(error^2)/(n-1))
        meanPlusSEM[i] = mean[i] + std::sqrt(meanPlusSEM[i]/(n[i]*(n[i]-1)));
}
