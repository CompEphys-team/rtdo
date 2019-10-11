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


#include "wavegenprogressplotter.h"
#include "ui_wavegenprogressplotter.h"

template <typename T>
struct GraphProxy : public AbstractGraphProxy
{
    GraphProxy(QVector<T> Wavegen::Archive::* data, QColor color, QCheckBox *cb) : AbstractGraphProxy(color, cb), data(data) {}
    ~GraphProxy() {}

    void populate()
    {
        QVector<QCPGraphData> vec((archive->*data).size());
        for ( int i = 0; i < vec.size(); i++ ) {
            vec[i].key = i;
            vec[i].value = (archive->*data).at(i);
        }
        dataPtr.reset(new QCPGraphDataContainer);
        extend();
    }

    void extend()
    {
        int size = dataPtr->size();
        QVector<QCPGraphData> vec((archive->*data).size() - size);
        for ( int i = 0; i < vec.size(); i++ ) {
            vec[i].key = i + size;
            vec[i].value = (archive->*data).at(i+size);
        }
        dataPtr->add(vec, true);
    }

    QVector<T> Wavegen::Archive::* data;
};

AbstractGraphProxy::AbstractGraphProxy(QColor color, QCheckBox *cb) : color(color), cb(cb)
{
    QPixmap pm(12,12);
    pm.fill(color);
    cb->setIcon(QIcon(pm));
}

WavegenProgressPlotter::WavegenProgressPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WavegenProgressPlotter),
    inProgress(false)
{
    ui->setupUi(this);

    proxies = {
        new GraphProxy<quint32>(&Wavegen::Archive::nElites, Qt::red, ui->nElites),
        new GraphProxy<quint32>(&Wavegen::Archive::nCandidates, Qt::green, ui->nCandidates),
        new GraphProxy<quint32>(&Wavegen::Archive::nInsertions, Qt::blue, ui->nInsertions),
        new GraphProxy<quint32>(&Wavegen::Archive::nReplacements, Qt::cyan, ui->nReplacements)
    };

    for ( size_t i = 0; i < proxies.size(); i++ ) {
        connect(proxies[i]->cb, &QCheckBox::toggled, [=](bool on){
            if ( ui->plot->graphCount() > int(i) ) {
                ui->plot->graph(i)->setVisible(on);
                ui->plot->replot();
            }
        });
    }

    ui->plot->yAxis->setLabel("Number of stimulations");
    ui->plot->xAxis->setLabel("Epoch");
    ui->plot->yAxis2->setLabel("Fitness");
    ui->plot->yAxis2->setVisible(true);
    ui->plot->xAxis->setRangeLower(0);
    ui->plot->yAxis->setRangeLower(0);
    ui->plot->yAxis2->setRangeLower(0);

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);
    connect(ui->plot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->plot->selectedAxes();
        if ( axes.isEmpty() )
           axes = ui->plot->axisRect()->axes();
        ui->plot->axisRect()->setRangeZoomAxes(axes);
        ui->plot->axisRect()->setRangeDragAxes(axes);
    });
    connect(ui->plot, &QCustomPlot::axisDoubleClick, this, [=](QCPAxis *axis, QCPAxis::SelectablePart, QMouseEvent*) {
        axis->rescale(true);
        axis->setRangeLower(0);
        ui->plot->replot();
    });
    ui->plot->axisRect()->setRangeZoomAxes(ui->plot->axisRect()->axes());
    ui->plot->axisRect()->setRangeDragAxes(ui->plot->axisRect()->axes());

    connect(ui->archive, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
}

void WavegenProgressPlotter::init(Session &session)
{
    this->session =& session;
    connect(&session.wavegen(), SIGNAL(done()), this, SLOT(updateArchives()));
    connect(&session.wavegen(), SIGNAL(searchTick(int)), this, SLOT(searchTick(int)));
    updateArchives();
}

WavegenProgressPlotter::~WavegenProgressPlotter()
{
    delete ui;
    for ( AbstractGraphProxy *proxy : proxies )
        delete proxy;
}

void WavegenProgressPlotter::updateArchives()
{
    int current = ui->archive->currentIndex();
    ui->archive->clear();
    QStringList labels;
    for ( size_t i = 0; i < session->wavegen().archives().size(); i++ ) {
        WaveSource src(*session, WaveSource::Archive, i);
        labels << src.prettyName();
    }
    ui->archive->addItems(labels);
    ui->archive->setCurrentIndex(current);
    inProgress = false;
}

void WavegenProgressPlotter::searchTick(int)
{
    const Wavegen::Archive &arch = session->wavegen().currentArchive();
    if ( !inProgress )
        ui->archive->addItem(QString("%1 (in progress)").arg(arch.action));
    else if ( ui->archive->currentIndex() == int(session->wavegen().archives().size()) ) {
        for ( AbstractGraphProxy *proxy : proxies )
            proxy->extend();
        ui->plot->replot();
    }
    inProgress = true;
}

void WavegenProgressPlotter::replot()
{
    ui->plot->clearGraphs();
    if ( ui->archive->currentIndex() < 0 )
        return;
    const Wavegen::Archive &arch = (ui->archive->currentIndex() < int(session->wavegen().archives().size()))
            ? session->wavegen().archives().at(ui->archive->currentIndex())
            : session->wavegen().currentArchive();

    QCPGraph *g;
    for ( AbstractGraphProxy *proxy : proxies ) {
        proxy->archive =& arch;
        proxy->populate();
        g = ui->plot->addGraph(proxy->xAxis, proxy->yAxis);
        g->setData(proxy->dataPtr);
        g->setPen(QPen(proxy->color));
        g->setVisible(proxy->cb->isChecked());
    }

    ui->plot->rescaleAxes(true);
    ui->plot->xAxis->setRangeLower(0);
    ui->plot->yAxis->setRangeLower(0);
    ui->plot->yAxis2->setRangeLower(0);
    ui->plot->replot();
}
