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
            if ( axis->axisType() == QCPAxis::atLeft )
                axis->setRange(p.min, p.max);
            else
                axis->rescale();
            plot->replot();
        });
        plot->axisRect()->setRangeZoomAxes(plot->axisRect()->axes());
        plot->axisRect()->setRangeDragAxes(plot->axisRect()->axes());
        plot->yAxis->setRange(p.min, p.max);
    }
    ui->columns->setMaximum(plots.size());
    setColumnCount(ui->columns->value());

    // Enslave to GAFitterWidget
    if ( enslave ) {
        ui->fits->setVisible(false);
        connect(&session->gaFitter(), &GAFitter::progress, this, &ParameterFitPlotter::progress);
        plots[0]->xAxis->setRange(0, 100);
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
        ui->fits->addItem(QString("Fit %1 (%2)").arg(i).arg(session->gaFitter().results().at(i).deck.prettyName()));
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
        QVector<double> values(fit.epochs);
        for ( quint32 epoch = 0; epoch < fit.epochs; epoch++ ) {
            values[epoch] = fit.params[epoch][i];
        }
        plots[i]->clearGraphs();
        QCPGraph *graph = plots[i]->addGraph();
        graph->setData(keys, values, true);
        plots[i]->rescaleAxes();
        plots[i]->replot();
    }
}

void ParameterFitPlotter::clear()
{
    for ( QCustomPlot *plot : plots ) {
        plot->clearGraphs();
        plot->addGraph();
        plot->replot();
    }
}

void ParameterFitPlotter::progress(quint32 epoch)
{
    const std::vector<scalar> &values = session->gaFitter().currentResults().params.at(epoch);
    for ( size_t i = 0; i < plots.size(); i++ ) {
        plots[i]->graph()->addData(epoch, values[i]);
    }

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
