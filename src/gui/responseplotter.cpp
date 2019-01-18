#include "responseplotter.h"
#include "ui_responseplotter.h"

ResponsePlotter::ResponsePlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ResponsePlotter),
    dataTimer(this)
{
    ui->setupUi(this);

    ui->plot->xAxis->setLabel("Time (ms)");
    ui->plot->xAxis->setRange(0, 10000);
    ui->plot->yAxis->setLabel("Voltage (mV)");
    ui->plot->yAxis->setRange(-100, 100);
    ui->plot->yAxis2->setLabel("Current (nA)");
    ui->plot->yAxis2->setRange(-1000, 1000);
    ui->plot->yAxis2->setVisible(true);

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
    connect(ui->plot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->plot->selectedAxes();
        if ( axes.isEmpty() )
           axes = ui->plot->axisRect()->axes();
        ui->plot->axisRect()->setRangeZoomAxes(axes);
        ui->plot->axisRect()->setRangeDragAxes(axes);

    });
    ui->plot->axisRect()->setRangeZoomAxes(ui->plot->axisRect()->axes());
    ui->plot->axisRect()->setRangeDragAxes(ui->plot->axisRect()->axes());

    connect(ui->plot, &QCustomPlot::axisDoubleClick, [=](QCPAxis *ax, QCPAxis::SelectablePart part){
        bool found;
        QCPRange keyRange = ui->plot->graph(1)->getKeyRange(found);
        QCPRange axRange = ui->plot->xAxis->range();
        if ( found && ax == ui->plot->xAxis ) {
            if ( part == QCPAxis::spAxisLabel )
                ax->setRange(keyRange);
            else
                ax->setRange(axRange + keyRange.upper - axRange.upper + (axRange.upper-axRange.lower)/20);
        }
    });

    connect(ui->Vcmd, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(0)->setVisible(on);
        ui->plot->replot();
    });
    connect(ui->Vresponse, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(1)->setVisible(on);
        ui->plot->replot();
    });
    connect(ui->V2, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(2)->setVisible(on);
        ui->plot->replot();
    });
    connect(ui->Iresponse, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(3)->setVisible(on);
        ui->plot->yAxis2->setVisible(on);
        ui->plot->replot();
    });

    ui->col_Vcmd->setColor(Qt::blue);
    ui->col_V1->setColor(Qt::red);
    ui->col_V2->setColor(Qt::darkRed);
    ui->col_I->setColor(Qt::darkGreen);
    connect(ui->col_Vcmd, &ColorButton::colorChanged, [=](QColor col){
        ui->plot->graph(0)->setPen(QPen(col));
        ui->plot->replot();
    });
    connect(ui->col_V1, &ColorButton::colorChanged, [=](QColor col){
        ui->plot->graph(1)->setPen(QPen(col));
        ui->plot->replot();
    });
    connect(ui->col_V2, &ColorButton::colorChanged, [=](QColor col){
        ui->plot->graph(2)->setPen(QPen(col));
        ui->plot->replot();
    });
    connect(ui->col_I, &ColorButton::colorChanged, [=](QColor col){
        ui->plot->graph(3)->setPen(QPen(col));
        ui->plot->replot();
    });

    clear();


    connect(&dataTimer, SIGNAL(timeout()), this, SLOT(replot()));
}

ResponsePlotter::~ResponsePlotter()
{
    delete ui;
}

void ResponsePlotter::setDAQ(DAQ *daq)
{
    this->daq = daq;
}

void ResponsePlotter::start()
{
    dataTimer.start(10);
}

void ResponsePlotter::stop()
{
    dataTimer.stop();
}

void ResponsePlotter::clear()
{
    ui->plot->clearGraphs();

    QCPGraph *g = ui->plot->addGraph();
    g->setPen(QPen(ui->col_Vcmd->color));
    g->setVisible(ui->Vcmd->isChecked());

    g = ui->plot->addGraph();
    g->setPen(QPen(ui->col_V1->color));
    g->setVisible(ui->Vresponse->isChecked());

    g = ui->plot->addGraph();
    g->setPen(QPen(ui->col_V2->color));
    g->setVisible(ui->V2->isChecked());

    g = ui->plot->addGraph(0, ui->plot->yAxis2);
    g->setPen(QPen(ui->col_I->color));
    g->setVisible(ui->Iresponse->isChecked());

    ui->plot->xAxis->moveRange(-ui->plot->xAxis->range().lower);

    iT = 0;

    ui->plot->replot();
}

void ResponsePlotter::replot()
{
    bool rangeFound;
    QCPRange range = ui->plot->graph(1)->getKeyRange(rangeFound);

    RTMaybe::Queue<DataPoint> *q[] = {&qO, &qV, &qV2, &qI};
    DataPoint point;

    if ( daq ) {
        double dt = daq->samplingDt();
        while ( daq->samplesRemaining > 0 ) {
            point.t = iT*dt;
            daq->next();
            try {
                ui->plot->graph(1)->addData(point.t, daq->voltage);
                ui->plot->graph(2)->addData(point.t, daq->voltage_2);
                ui->plot->graph(3)->addData(point.t, daq->current);
            } catch ( std::bad_alloc ) {
                double middle = range.lower + (range.upper - range.lower) / 2;
                for ( int j = 1; j < 4; j++ )
                    ui->plot->graph(j)->data()->removeBefore(middle);
                ui->plot->graph(1)->addData(point.t, daq->voltage);
                ui->plot->graph(2)->addData(point.t, daq->voltage_2);
                ui->plot->graph(3)->addData(point.t, daq->current);
            }
            ++iT;
        }
    } else {
        for ( int i = 0; i < 4; i++ ) {
            while ( q[i]->pop_if(point) ) {
                try {
                    ui->plot->graph(i)->addData(point.t, point.value);
                } catch ( std::bad_alloc ) {
                    // Out of memory - remove half of all data to keep going
                    double middle = range.lower + (range.upper - range.lower) / 2;
                    for ( int j = 0; j < 4; j++ )
                        ui->plot->graph(j)->data()->removeBefore(middle);
                    // Then try again
                    ui->plot->graph(i)->addData(point.t, point.value);
                }
            }
        }
    }

    if ( rangeFound ) {
        double xUpper = ui->plot->xAxis->range().upper;
        if ( range.upper <= xUpper && point.t > xUpper) {
            ui->plot->xAxis->moveRange(point.t - xUpper);
        }
    }

    ui->plot->replot();
}
