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

    connect(ui->Vcmd, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(0)->setVisible(on);
    });
    connect(ui->Vresponse, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(1)->setVisible(on);
    });
    connect(ui->Iresponse, &QCheckBox::toggled, [=](bool on){
        ui->plot->graph(2)->setVisible(on);
    });

    clear();

    connect(&dataTimer, SIGNAL(timeout()), this, SLOT(replot()));
    dataTimer.start(20);
}

ResponsePlotter::~ResponsePlotter()
{
    delete ui;
}

void ResponsePlotter::clear()
{
    ui->plot->clearGraphs();

    QCPGraph *g = ui->plot->addGraph();
    g->setPen(QPen(Qt::blue));
    g->setVisible(ui->Vcmd->isChecked());

    g = ui->plot->addGraph();
    g->setPen(QPen(Qt::red));
    g->setVisible(ui->Vresponse->isChecked());

    g = ui->plot->addGraph(0, ui->plot->yAxis2);
    g->setPen(QPen(Qt::darkGreen));
    g->setVisible(ui->Iresponse->isChecked());

    ui->plot->xAxis->moveRange(-ui->plot->xAxis->range().lower);
}

void ResponsePlotter::replot()
{
    bool rangeFound;
    QCPRange range = ui->plot->graph()->getKeyRange(rangeFound);

    RTMaybe::Queue<DataPoint> *q[3] = {&qO, &qV, &qI};
    DataPoint point;

    for ( int i = 0; i < 3; i++ ) {
        while ( q[i]->pop_if(point) ) {
            ui->plot->graph(i)->addData(point.t, point.value);
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
