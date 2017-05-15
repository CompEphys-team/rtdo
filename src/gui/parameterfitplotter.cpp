#include "parameterfitplotter.h"
#include "ui_parameterfitplotter.h"

ParameterFitPlotter::ParameterFitPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ParameterFitPlotter),
    resizing(false)
{
    ui->setupUi(this);
    ui->table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(setColumnCount(int)));
    connect(ui->table->verticalHeader(), SIGNAL(sectionResized(int,int,int)), this, SLOT(resizeTableRows(int,int,int)));
    connect(ui->param, &QCheckBox::stateChanged, [=](int state) {
        bool on = state == Qt::Checked;
        for ( QCustomPlot *p : plots ) {
            p->graph(0)->setVisible(on);
            p->item(0)->setVisible(on); // This should be the target value line
            p->yAxis->setTicks(on);
            p->yAxis->setTickLabels(on);
            p->replot();
        }
    });
    connect(ui->error, &QCheckBox::stateChanged, [=](int state) {
        bool on = state == Qt::Checked;
        for ( QCustomPlot *p : plots ) {
            p->graph(1)->setVisible(on);
            p->yAxis2->setVisible(on);
            p->replot();
        }
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
    for ( QCustomPlot *p : plots )
        delete p;
}

void ParameterFitPlotter::init(Session *session, bool enslave)
{
    this->session = session;

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
        ui->fits->setVisible(false);
        ui->rescale->setVisible(false);
        connect(&session->gaFitter(), &GAFitter::progress, this, &ParameterFitPlotter::progress);
        plots[0]->xAxis->setRange(0, 100);
        clear();
    } else {
        connect(&session->gaFitter(), SIGNAL(done()), this, SLOT(updateFits()));
        connect(ui->fits, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
        updateFits();
    }
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
    int currentFit = ui->fits->currentIndex();
    ui->fits->clear();
    for ( size_t i = 0; i < session->gaFitter().results().size(); i++ ) {
        const GAFitter::Output &fit = session->gaFitter().results().at(i);
        ui->fits->addItem(QString("Fit %1 (%2 epochs, %3)").arg(i).arg(fit.epochs).arg(fit.deck.prettyName()));
    }
    ui->fits->setCurrentIndex(currentFit < 0 ? 0 : currentFit);
}

void ParameterFitPlotter::replot()
{
    int currentFit = ui->fits->currentIndex();
    if ( currentFit < 0 )
        return;
    const GAFitter::Output fit = session->gaFitter().results().at(currentFit);
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
        plots[i]->clearGraphs();
        QCPGraph *graph = plots[i]->addGraph();
        graph->setData(keys, values, true);
        graph->setVisible(ui->param->isChecked());
        if ( ui->rescale->isChecked() ) {
            plots[i]->xAxis->rescale();
            plots[i]->yAxis->rescale();
        }

        QCPGraph *errGraph = plots[i]->addGraph(0, plots[i]->yAxis2);
        errGraph->setData(errKey, errors, true);
        errGraph->setVisible(ui->error->isChecked());
        styleErrorGraph(errGraph);
        if ( ui->rescale->isChecked() ) {
            if ( i == 0 )
                plots[i]->yAxis2->setRange(0,1); // Reset range when selecting a new fit
            bool found;
            QCPRange range = errGraph->getValueRange(found);
            if ( found && range.upper > plots[i]->yAxis2->range().upper )
                plots[i]->yAxis2->setRangeUpper(range.upper);
        }

        plots[i]->replot();
    }
}

void ParameterFitPlotter::styleErrorGraph(QCPGraph *g)
{
    static QPen *pen = nullptr;
    if ( !pen ) {
        pen = new QPen();
        pen->setColor(Qt::magenta);
    }
    g->setLineStyle(QCPGraph::lsStepLeft);
    g->setPen(*pen);
}

void ParameterFitPlotter::clear()
{
    for ( QCustomPlot *plot : plots ) {
        plot->clearGraphs();
        plot->addGraph()->setVisible(ui->param->isChecked());
        plot->addGraph(0, plot->yAxis2)->setVisible(ui->error->isChecked());
        styleErrorGraph(plot->graph(1));
        plot->replot();
    }
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
