#include "parameterfitplotter.h"
#include "ui_parameterfitplotter.h"
#include "colorbutton.h"
#include "rundatadialog.h"
#include "daqdialog.h"
#include "gafittersettingsdialog.h"
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
        if ( summarising )
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

    connect(ui->fits, &QTableWidget::itemSelectionChanged, [=]() {
        QList<QTableWidgetSelectionRange> rlist = ui->fits->selectedRanges();
        ui->settingsButtons->setEnabled(rlist.size() == 1 && rlist.first().rowCount() == 1);
    });

    connect(ui->rundata, &QPushButton::clicked, [=](bool){
        std::vector<int> rows = getSelectedRows(ui->fits);
        if ( rows.size() != 1 )
            return;
        RunDataDialog *dlg = new RunDataDialog(*session, session->gaFitter().results().at(rows[0]).resultIndex);
        dlg->setWindowTitle(QString("%1 for fit %2").arg(dlg->windowTitle()).arg(rows[0]));
        dlg->show();
    });
    connect(ui->daqdata, &QPushButton::clicked, [=](bool){
        std::vector<int> rows = getSelectedRows(ui->fits);
        if ( rows.size() != 1 )
            return;
        DAQDialog *dlg = new DAQDialog(*session, session->gaFitter().results().at(rows[0]).resultIndex);
        dlg->setWindowTitle(QString("%1 for fit %2").arg(dlg->windowTitle()).arg(rows[0]));
        dlg->show();
    });
    connect(ui->fittersettings, &QPushButton::clicked, [=](bool){
        std::vector<int> rows = getSelectedRows(ui->fits);
        if ( rows.size() != 1 )
            return;
        GAFitterSettingsDialog *dlg = new GAFitterSettingsDialog(*session, session->gaFitter().results().at(rows[0]).resultIndex);
        dlg->setWindowTitle(QString("%1 for fit %2").arg(dlg->windowTitle()).arg(rows[0]));
        dlg->show();
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
        ui->sidebar->setVisible(false);
        ui->legend->setVisible(false);
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
        ui->groups->setColumnWidth(1, 80);
        ui->groups->horizontalHeader()->setFixedHeight(5);
        ui->groups->verticalHeader()->setSectionsMovable(true);
        connect(&session->gaFitter(), SIGNAL(done()), this, SLOT(updateFits()));
        connect(ui->fits, &QTableWidget::itemSelectionChanged, [=]{
            summarising = false;
            replot();
        });
        connect(ui->groups, &QTableWidget::itemSelectionChanged, [=]{
            summarising = true;
            replot();
        });
        connect(ui->boxplot_epoch, SIGNAL(valueChanged(int)), this, SLOT(reBoxPlot()));
        updateFits();
    }
    QTimer::singleShot(10, this, &ParameterFitPlotter::resizePanel);
}

void ParameterFitPlotter::updateFits()
{
    int col0 = 11;
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
        ui->fits->setItem(i, 2, new QTableWidgetItem(fit.stimSource.prettyName()));
        ui->fits->setItem(i, 3, new QTableWidgetItem(QString::number(fit.epochs)));
        ui->fits->setItem(i, 4, new QTableWidgetItem(QString::number(session->gaFitterSettings(fit.resultIndex).randomOrder)));
        ui->fits->setItem(i, 5, new QTableWidgetItem(QString::number(session->gaFitterSettings(fit.resultIndex).crossover, 'g', 2)));
        ui->fits->setItem(i, 6, new QTableWidgetItem(session->gaFitterSettings(fit.resultIndex).decaySigma ? "Y" : "N"));
        ui->fits->setItem(i, 8, new QTableWidgetItem(session->gaFitterSettings(fit.resultIndex).useDE ? "DE" : "GA"));
        ui->fits->setItem(i, 9, new QTableWidgetItem(session->gaFitterSettings(fit.resultIndex).useClustering ? "Y" : "N"));
        ui->fits->setItem(i, 10, new QTableWidgetItem(QString::number(session->gaFitterSettings(fit.resultIndex).mutationSelectivity)));
        if ( session->daqData(fit.resultIndex).simulate == -1 ) {
            ui->fits->setItem(i, 7, new QTableWidgetItem(fit.VCRecord));
            for ( size_t j = 0; j < session->project.model().adjustableParams.size(); j++ )
                ui->fits->setItem(i, col0+j, new QTableWidgetItem(QString::number(fit.targets[j], 'g', 3)));
        } else if ( session->daqData(fit.resultIndex).simulate == 0 ) {
            ui->fits->setItem(i, 7, new QTableWidgetItem(QString("live DAQ")));
        } else if ( session->daqData(fit.resultIndex).simulate == 1 ) {
            ui->fits->setItem(i, 7, new QTableWidgetItem(QString("%1-%2").arg(session->daqData(fit.resultIndex).simulate).arg(session->daqData(fit.resultIndex).simd.paramSet)));
            for ( size_t j = 0; j < session->project.model().adjustableParams.size(); j++ )
                ui->fits->setItem(i, col0+j, new QTableWidgetItem(QString::number(fit.targets[j], 'g', 3)));
        }
        if ( fit.final )
            for ( size_t j = 0, np = session->project.model().adjustableParams.size(); j < np; j++ )
                ui->fits->setItem(i, col0+np+1+j, new QTableWidgetItem(QString::number(fit.finalParams[j], 'g', 3)));
    }
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

    buildPlotLayout();

    reBoxPlot();
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
    bool initial = true;
    std::vector<int> rows = getSelectedRows(ui->fits);
    if ( rows.empty() )
        return;

    ui->panel->layer("target")->setVisible(ui->param->isChecked());

    for ( int row : rows ) {
        const GAFitter::Output fit = session->gaFitter().results().at(row);
        QVector<double> keys(fit.epochs);
        for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ )
            keys[epoch] = epoch;
        for ( size_t i = 0; i < axRects.size(); i++ ) {
            QCPAxis *xAxis = axRects[i]->axis(QCPAxis::atBottom);
            QCPAxis *yAxis = axRects[i]->axis(QCPAxis::atLeft, 0);
            QCPAxis *yAxis2 = axRects[i]->axis(QCPAxis::atRight);

            QCPItemStraightLine *line = new QCPItemStraightLine(ui->panel);
            QPen pen(getGraphColorBtn(row)->color);
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
            for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ ) {
                values[epoch] = fit.params[epoch][i];
                if ( fit.targetParam[epoch] == i ) {
                    errors.push_back(fit.error[epoch]);
                    errKey.push_back(epoch);
                }
            }

            QColor col(getGraphColorBtn(row)->color);
            col.setAlphaF(ui->opacity->value()/100.);
            QCPGraph *graph = addGraph(xAxis, yAxis, col, keys, values, "main", ui->param->isChecked());
            if ( ui->rescale->isChecked() ) {
                xAxis->rescale();
                yAxis->rescale();
            }

            QColor errCol(getErrorColorBtn(row)->color);
            errCol.setAlphaF(ui->opacity->value()/100.);
            QCPGraph *errGraph = addGraph(xAxis, yAxis2, errCol, errKey, errors, "main", ui->error->isChecked());
            errGraph->setLineStyle(QCPGraph::lsStepLeft);
            if ( ui->rescale->isChecked() ) {
                if ( initial ) {
                    yAxis2->setRange(0,1); // Reset range when selecting a new fit
                    initial = false;
                }
                bool found;
                QCPRange range = errGraph->getValueRange(found);
                if ( found && range.upper > yAxis2->range().upper )
                    yAxis2->setRangeUpper(range.upper);
            }
            if ( session->gaFitterSettings(fit.resultIndex).useLikelihood )
                yAxis2->setLabel("-log likelihood");
            else
                yAxis2->setLabel("RMS Error (nA)");

            if ( i == 0 ) {
                graph->setName(QString("Fit %1").arg(row));
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
    axRects[fit.targetParam[epoch]]->axis(QCPAxis::atRight)->graphs().first()->addData(epoch, fit.error[epoch]);

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

void ParameterFitPlotter::addGroup(std::vector<int> group, QString label)
{
    if ( group.empty() ) {
        group = getSelectedRows(ui->fits);
        if ( group.empty() )
            return;
        std::sort(group.begin(), group.end());
    }
    groups.push_back(group);

    int row = ui->groups->rowCount();
    ui->groups->insertRow(row);
    ColorButton *c = new ColorButton();
    c->setColor(QColorDialog::standardColor(row % 20));
    ui->groups->setCellWidget(row, 0, c);
    connect(c, SIGNAL(colorChanged(QColor)), this, SLOT(replot()));

    QString numbers;
    for ( int i = 0, last = group.size()-1; i <= last; i++ ) {
        int beginning = group[i];
        while ( i < last && group[i+1] == group[i] + 1 )
            ++i;
        if ( group[i] > beginning )
            numbers.append(QString("%1-%2").arg(beginning).arg(group[i]));
        else
            numbers.append(QString::number(group[i]));
        if ( i < last )
            numbers.append("; ");
    }
    ui->groups->setItem(row, 1, new QTableWidgetItem(numbers));

    QTableWidgetItem *item = new QTableWidgetItem(label);
    ui->groups->setItem(row, 2, item);
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

void ParameterFitPlotter::on_saveGroups_clicked()
{
    if ( groups.empty() )
        return;
    QString file = QFileDialog::getSaveFileName(this, "Save groups to file...", session->directory());
    if ( file.isEmpty() )
        return;
    std::ofstream os(file.toStdString());
    for ( size_t vi = 0; vi < groups.size(); vi++ ) {
        int i = ui->groups->verticalHeader()->logicalIndex(vi);
        os << groups[i].size() << ':';
        for ( int f : groups[i] )
            os << f << ',';
        os << ui->groups->item(i, 2)->text().toStdString() << std::endl;
    }
}

void ParameterFitPlotter::on_loadGroups_clicked()
{
    QString file = QFileDialog::getOpenFileName(this, "Select saved groups file...", session->directory());
    if ( file.isEmpty() )
        return;
    std::ifstream is(file.toStdString());
    int size;
    std::vector<int> group;
    char tmp;
    std::string label;
    is >> size;
    while ( is.good() ) {
        is >> tmp;
        group.resize(size);
        for ( int i = 0; i < size; i++ )
            is >> group[i] >> tmp;
        std::getline(is, label);
        addGroup(group, QString::fromStdString(label));
        is >> size;
    }
}

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
    std::vector<int> rows = getSelectedRows(ui->groups);
    if ( rows.empty() )
        return;

    ui->panel->layer("target")->setVisible(false);

    Filter *filter = nullptr;
    if ( ui->filter->value() > 1 )
        filter = new Filter(FilterMethod::SavitzkyGolayEdge5, 2*int(ui->filter->value()/2) + 1);

    for ( int row : rows ) {
        quint32 epochs = 0;
        bool hasFinal = true;
        for ( int fit : groups[row] ) {
            epochs = std::max(session->gaFitter().results().at(fit).epochs, epochs);
            hasFinal &= session->gaFitter().results().at(fit).final;
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
                getSummary(groups[row], [=](const GAFitter::Output &fit, int ep){
                    // Parameter error relative to target (%) :
                    return 100 * std::fabs(1 - fit.params[ep][i] / fit.targets[i]);
                }, mean, sem, median, max, filter);
                if ( hasFinal )
                    getSummary(groups[row], [=](const GAFitter::Output &fit, int){
                        return 100 * std::fabs(1 - fit.finalParams[i] / fit.targets[i]);
                    }, fMean, fSem, fMedian, fMax);
            } else {
                getSummary(groups[row], [=](const GAFitter::Output &fit, int ep){
                    // Parameter error relative to range (%) :
                    double range = session->gaFitterSettings(fit.resultIndex).constraints[i] == 1
                            ? (session->gaFitterSettings(fit.resultIndex).max[i] - session->gaFitterSettings(fit.resultIndex).min[i])
                            : (p.max - p.min);
                    return 100 * std::fabs((fit.params[ep][i] - fit.targets[i]) / range);
                }, mean, sem, median, max, filter);
                if ( hasFinal )
                    getSummary(groups[row], [=](const GAFitter::Output &fit, int){
                        double range = session->gaFitterSettings(fit.resultIndex).constraints[i] == 1
                                ? (session->gaFitterSettings(fit.resultIndex).max[i] - session->gaFitterSettings(fit.resultIndex).min[i])
                                : (p.max - p.min);
                        return 100 * std::fabs((fit.finalParams[i] - fit.targets[i]) / range);
                    }, fMean, fSem, fMedian, fMax);
            }
            getSummary(groups[row], [=](const GAFitter::Output &fit, int ep) -> double {
                for ( ; ep >= 0; ep-- )
                    if ( fit.targetParam[ep] == i )
                        return fit.error[ep];
                return 0;
            }, errMean, errSEM, errMedian, errMax, filter);
            if ( hasFinal )
                getSummary(groups[row], [=](const GAFitter::Output &fit, int){
                    return fit.finalError[i];
                }, fErrMean, fErrSEM, fErrMedian, fErrMax);

            QColor col(getGroupColorBtn(row)->color);
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
                if ( ui->groups->item(row, 2)->text().isEmpty() )
                    graph->setName(ui->groups->item(row, 1)->text());
                else
                    graph->setName(ui->groups->item(row, 2)->text());
                errGraph->setName("");
                graph->addToLegend();
                errGraph->addToLegend();
            }
        }
    }
}

void ParameterFitPlotter::getSummary(std::vector<int> fits,
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


// Interpolated quartiles -- https://stackoverflow.com/a/37708864
template<typename T>
static inline double Lerp(T v0, T v1, T t)
{
    return (1 - t)*v0 + t*v1;
}

template<typename T>
static inline std::vector<T> Quantile(const std::vector<T>& inData, const std::vector<T>& probs)
{
    if (inData.empty())
    {
        return std::vector<T>();
    }

    if (1 == inData.size())
    {
        return std::vector<T>(1, inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        T poi = Lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = Lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

void ParameterFitPlotter::reBoxPlot()
{
    ui->boxplot->clearPlottables();

    std::vector<std::vector<int>> selection;
    std::vector<QColor> colors;
    std::vector<QString> labels;
    if ( summarising ) {
        std::vector<int> rows = getSelectedRows(ui->groups);
        for ( int row : rows ) {
            selection.push_back(groups[row]);
            colors.push_back(getGroupColorBtn(row)->color);
            labels.push_back(ui->groups->item(row, 2)->text().isEmpty()
                             ? ui->groups->item(row, 1)->text()
                             : ui->groups->item(row, 2)->text());
        }
    } else {
        std::vector<int> rows = getSelectedRows(ui->fits);
        selection.push_back(rows);
        colors.push_back(QColor(Qt::black));
        labels.push_back("");
    }

    if ( selection.empty() || (selection.size() == 1 && selection[0].size() < 2) ) {
        ui->boxplot->replot();
        return;
    }

    // Find any parameters that aren't fitted at all
    std::vector<bool> fitted(session->project.model().adjustableParams.size(), false);
    int nFitted = 0;
    for ( size_t i = 0; i < selection.size(); i++ )
        for ( int f : selection[i] )
            for ( size_t j = 0; j < fitted.size(); j++ )
                fitted[j] = session->gaFitterSettings(session->gaFitter().results().at(f).resultIndex).constraints[j] < 2;
    for ( const bool &b : fitted )
        nFitted += b;

    // Set up ticks
    int stride = selection.size() + 1;
    int tickOffset = selection.size() / 2;
    double barOffset = selection.size()%2 ? 0 : 0.5;
    ui->boxplot->xAxis->setSubTicks(false);
    ui->boxplot->xAxis->setTickLength(0, 4);
    ui->boxplot->xAxis->grid()->setVisible(false);
    QSharedPointer<QCPAxisTickerText> textTicker(new QCPAxisTickerText);
    for ( size_t i = 0, fi = 0; i < fitted.size(); fi += fitted[i], i++ ) {
        const AdjustableParam &p = session->project.model().adjustableParams[i];
        if ( fitted[i] )
            textTicker->addTick(tickOffset + fi*stride, QString::fromStdString(p.name) + (p.multiplicative ? "¹" : "²"));
    }
    textTicker->addTick(tickOffset + nFitted*stride, "joint");
    ui->boxplot->xAxis->setTicker(textTicker);
    ui->boxplot->yAxis->setLabel("Deviation (¹ %, ² % range)");
    ui->boxplot->legend->setVisible(summarising);

    // Enter data group by group
    quint32 epoch = ui->boxplot_epoch->value();
    for ( size_t i = 0; i < selection.size(); i++ ) {
        QCPStatisticalBox *box = new QCPStatisticalBox(ui->boxplot->xAxis, ui->boxplot->yAxis);
        box->setName(labels[i]);
        QPen whiskerPen(Qt::SolidLine);
        whiskerPen.setCapStyle(Qt::FlatCap);
        box->setWhiskerPen(whiskerPen);
        box->setPen(QPen(colors[i]));
        colors[i].setAlphaF(0.3);
        box->setBrush(QBrush(colors[i]));
        box->setWidth(0.8);

        std::vector<std::vector<double>> outcomes(nFitted + 1);
        for ( int f : selection[i] ) {
            const GAFitter::Output &fit = session->gaFitter().results().at(f);
            const std::vector<scalar> *params;
            if ( fit.final && (epoch == 0 || epoch >= fit.epochs) )
                params =& fit.finalParams;
            else if ( epoch >= fit.epochs )
                params =& fit.params[fit.epochs];
            else
                params =& fit.params[epoch];
            double value, total = 0;
            for ( size_t j = 0, fj = 0; j < fitted.size(); fj += fitted[j], j++ ) {
                if ( !fitted[j] )
                    continue;
                const AdjustableParam &p = session->project.model().adjustableParams.at(j);
                if ( p.multiplicative ) {
                    value = 100 * std::fabs(1 - params->at(j) / fit.targets[j]);
                } else {
                    double range = session->gaFitterSettings(fit.resultIndex).constraints[j] == 1
                            ? (session->gaFitterSettings(fit.resultIndex).max[j] - session->gaFitterSettings(fit.resultIndex).min[j])
                            : (p.max - p.min);
                    value = 100 * std::fabs((params->at(j) - fit.targets[j]) / range);
                }
                total += value;
                outcomes[fj].push_back(value);
            }
            outcomes.back().push_back(total/nFitted);
        }
        for ( size_t j = 0; j < outcomes.size(); j++ ) {
            std::vector<double> q = Quantile(outcomes[j], {0, 0.25, 0.5, 0.75, 1});
            box->addData(j*stride + i + barOffset, q[0], q[1], q[2], q[3], q[4]);
        }
    }

    ui->boxplot->rescaleAxes();
    ui->boxplot->xAxis->setRange(-1, (nFitted+1)*stride - 1);

    ui->boxplot->replot();
}

void ParameterFitPlotter::on_boxplot_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->boxplot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), file);
}
