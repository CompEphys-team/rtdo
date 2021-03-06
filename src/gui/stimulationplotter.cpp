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


#include "stimulationplotter.h"
#include "ui_stimulationplotter.h"
#include "colorbutton.h"
#include <QTimer>

StimulationPlotter::StimulationPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StimulationPlotter),
    rebuilding(false)
{
    ui->setupUi(this);
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(replot()));
    connect(ui->sources, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->nStims, SIGNAL(valueChanged(int)), this, SLOT(replot()));
    connect(ui->offset, SIGNAL(valueChanged(int)), this, SLOT(replot()));
    connect(ui->tails, SIGNAL(toggled(bool)), this, SLOT(replot()));
    connect(ui->slider, SIGNAL(valueChanged(int)), this, SLOT(resizePanel()));

    connect(ui->legend, &QTableWidget::cellChanged, [=](int row, int col){ // Show/hide overlay graphs by legend column 0 checkstate
        if ( !rebuilding && col == 0 ) {
            bool on = ui->legend->item(row, col)->checkState() == Qt::Checked;
            ui->overlay->graph(row)->setVisible(on);
            ui->overlay->replot();
        }
    });
    connect(ui->legend->horizontalHeader(), &QHeaderView::sectionClicked, [=](int idx) {
        static bool on = false;
        if ( idx == 0 ) {
            rebuilding = true;
            for ( int i = 0; i < ui->legend->rowCount(); i++ ) {
                ui->legend->item(i, 0)->setCheckState(on ? Qt::Checked : Qt::Unchecked);
                ui->overlay->graph(i)->setVisible(on);
            }
            on = !on;
            ui->overlay->replot();
            rebuilding = false;
        }
    });

    connect(ui->scale, &QCheckBox::toggled, [=](bool on) {
        for ( QCPGraph *g : graphs ) {
            g->keyAxis()->setTicks(on);
            g->keyAxis()->setTickLabels(on);
            g->keyAxis()->setLabel(on ? "Time (ms)" : "");
            g->valueAxis()->setTicks(on);
            g->valueAxis()->setTickLabels(on);
            g->valueAxis()->setLabel(on ? "Voltage (mV)" : "");
        }
        ui->panel->replot();
    });
    connect(ui->titles, &QCheckBox::toggled, [=](bool on) {
        QCPLayoutGrid *grid = ui->panel->plotLayout();
        for ( int row = 0; row < grid->rowCount(); row += 2 )
            for ( int col = 0; col < grid->columnCount() && (row+2)/2*(col+1) <= int(graphs.size()); col++ )
                grid->element(row, col)->setVisible(on);
        ui->panel->replot();
    });

    ui->panel->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);

    ui->overlay->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->overlay->xAxis->setLabel("Time (ms)");
    ui->overlay->yAxis->setLabel("Voltage (mV)");

    ui->legend->setColumnWidth(0, 21);
    ui->legend->setColumnWidth(1, 25);
    ui->splitter->setStretchFactor(0, 4);
    ui->splitter->setStretchFactor(1, 4);
    ui->splitter->setStretchFactor(2, 1);
}

StimulationPlotter::StimulationPlotter(Session &session, QWidget *parent) :
    StimulationPlotter(parent)
{
    init(&session);
}

StimulationPlotter::~StimulationPlotter()
{
    delete ui;
}

void StimulationPlotter::resizeEvent(QResizeEvent *event)
{
    QWidget::resizeEvent(event);
    resizePanel();
}

void StimulationPlotter::resizePanel()
{
    double height = std::max(1, ui->slider->height() * ui->slider->value() / ui->slider->maximum());
    int nRows = (graphs.size() + ui->columns->value() - 1) / ui->columns->value();
    ui->panel->setFixedHeight(height * nRows);
}

void StimulationPlotter::init(Session *session)
{
    this->session = session;
    connect(&session->wavesets(), SIGNAL(addedSet()), this, SLOT(updateSources()));

    QStringList legendHeaderLabels;
    int nFixedLegendColumns = ui->legend->columnCount();
    for ( int i = 0; i < nFixedLegendColumns; i++ )
        legendHeaderLabels << ui->legend->horizontalHeaderItem(i)->text();
    for ( const AdjustableParam &p : session->project.model().adjustableParams )
        legendHeaderLabels << QString::fromStdString(p.name);
    ui->legend->setColumnCount(nFixedLegendColumns + session->project.model().adjustableParams.size());
    ui->legend->setHorizontalHeaderLabels(legendHeaderLabels);

    updateSources();
    QTimer::singleShot(10, this, &StimulationPlotter::resizePanel);
}

void StimulationPlotter::clear()
{
    ui->sources->setVisible(true);
    updateSources();
}

void StimulationPlotter::updateSources()
{
    int currentSource = ui->sources->currentIndex();
    ui->sources->clear();
    session->wavesets().selections();
    for ( WaveSource const& s : session->wavesets().sources() ) {
        ui->sources->addItem(s.prettyName());
    }
    ui->sources->setCurrentIndex(currentSource < 0 ? 0 : currentSource);
}

void StimulationPlotter::replot()
{
    if ( rebuilding || ui->sources->currentIndex() < 0 )
        return;

    const RunData &rd = session->qRunData();
    double duration = 0, minV = session->qStimulationData().minVoltage, maxV = session->qStimulationData().maxVoltage;
    WaveSource src = session->wavesets().sources().at(ui->sources->currentIndex());
    std::vector<Stimulation> stims = src.stimulations();
    std::vector<iObservations> obs = src.observations(rd.dt);
    std::vector<MAPElite> elites = src.elites();
    if ( stims.empty() )
        return;

    rebuilding = true;

    ui->nStims->setMaximum(stims.size());
    ui->offset->setMaximum(stims.size()-1);
    ui->nStims->setSuffix(QString("/%1").arg(stims.size()));
    ui->offset->setSingleStep(ui->nStims->value());
    size_t lower = ui->offset->value();
    size_t upper = std::min(lower + ui->nStims->value(), stims.size());
    ui->columns->setMaximum(upper-lower);

    // Legend
    for ( size_t i = colors.size(); i < upper-lower; i++ ) {
        colors.push_back(QColorDialog::standardColor(i % 42));
    }
    ui->legend->clearContents();
    ui->legend->setRowCount(upper-lower);
    QStringList labels;
    QTableWidgetItem check;
    check.setFlags(Qt::ItemIsUserCheckable | Qt::ItemIsEnabled);
    check.setCheckState(Qt::Checked);
    for ( size_t i = lower, row = 0; i < upper; i++, row++ ) {
        if ( src.type == WaveSource::Deck )
            labels << QString::fromStdString(session->project.model().adjustableParams[i].name);
        else
            labels << QString::number(i);

        ui->legend->setItem(row, 0, new QTableWidgetItem(check));

        ColorButton *btn = new ColorButton();
        btn->setColor(colors[row]);
        ui->legend->setCellWidget(row, 1, btn);
        connect(btn, &ColorButton::colorChanged, [=](QColor color){
            colors[row] = color;
            updateColor(row, true);
        });

        if ( !elites[i].deviations.empty() ) {
            ui->legend->setItem(row, 2, new QTableWidgetItem(QString::number(elites[i].fitness)));
            ui->legend->setItem(row, 3, new QTableWidgetItem(QString::number(elites[i].current)));
            for ( size_t j = 0; j < session->project.model().adjustableParams.size(); j++ )
                ui->legend->setItem(row, 4 + j, new QTableWidgetItem(QString::number(elites[i].deviations[j])));
        }
    }
    ui->legend->setVerticalHeaderLabels(labels);

    // Plots
    graphs.resize(upper-lower);
    ui->overlay->clearGraphs();
    ui->overlay->clearItems();

    ui->panel->clearPlottables();
    ui->panel->plotLayout()->clear();

    bool hasTitle = ui->titles->isChecked();
    bool hasScale = ui->scale->isChecked();
    int fig_col = ui->fig_column->currentIndex();

    for ( size_t i = 0, end = upper-lower; i < end; i++ ) {
        int row = 2 * int(i / ui->columns->value());
        int col = i % ui->columns->value();

        StimulationGraph *g = new StimulationGraph(ui->overlay->xAxis, ui->overlay->yAxis, stims[i+lower]); // ui->overlay takes ownership
        g->setObservations(obs[i+lower], rd.dt);
        QCPAxisRect *axes = new QCPAxisRect(ui->panel);
        QCPAxis *xAxis = axes->axis(QCPAxis::atBottom);
        QCPAxis *yAxis = axes->axis(QCPAxis::atLeft);
        xAxis->setTicks(hasScale);
        xAxis->setTickLabels(hasScale && (fig_col==0 || i >= end-ui->columns->value()));
        xAxis->setLabel((hasScale && (fig_col==0 || i >= end-ui->columns->value())) ? "Time (ms)" : "");
        xAxis->setLayer("axes");
        xAxis->grid()->setLayer("grid");
        xAxis->grid()->setVisible(false);
        yAxis->setTicks(hasScale);
        yAxis->setTickLabels(hasScale && (fig_col==0 || (fig_col == 1 && col == 0)));
        yAxis->setLabel((hasScale && (fig_col==0 || (fig_col == 1 && col == 0))) ? "Voltage (mV)" : "");
        yAxis->setLayer("axes");
        yAxis->grid()->setLayer("grid");
        yAxis->grid()->setVisible(false);

        QCPTextElement *title = new QCPTextElement(ui->panel, labels.at(i));
        title->setVisible(hasTitle);
        ui->panel->plotLayout()->addElement(row, col, title);
        ui->panel->plotLayout()->addElement(row+1, col, axes); // add axes to panel

        graphs[i] = new StimulationGraph(xAxis, yAxis, stims[i+lower], !ui->tails->isChecked()); // add new stimGraph to axes
        graphs[i]->setObservations(obs[i+lower], rd.dt);

        duration = std::max(duration, double(stims[i+lower].duration));
        updateColor(i, false);
    }
    ui->overlay->xAxis->setRange(0, duration);
    ui->overlay->yAxis->setRange(minV, maxV);
    ui->overlay->replot();

    for ( QCPGraph *g : graphs ) {
        g->keyAxis()->setRange(0, duration);
        g->valueAxis()->setRange(minV, maxV);
    }
    ui->panel->replot();
    resizePanel();

    rebuilding = false;
}

void StimulationPlotter::updateColor(size_t idx, bool replot)
{
    QColor c = colors[idx];
    QPen pen(c);
    pen.setWidth(2);
    c.setAlpha(c.alpha()/5.0);
    QBrush brush(c);
    ui->overlay->graph(idx)->setPen(pen);
    ui->overlay->graph(idx)->setBrush(brush);
    if ( replot )
        ui->overlay->replot();
    graphs[idx]->setPen(pen);
    graphs[idx]->setBrush(brush);
    if ( replot )
        ui->panel->replot();
}

void StimulationPlotter::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");

    ui->panel->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), ui->sources->currentText());
}
