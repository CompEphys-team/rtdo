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


#include "pcaplot.h"
#include "ui_pcaplot.h"
#include "session.h"
#include "populationsaver.h"

PCAPlot::PCAPlot(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PCAPlot),
    data(QSharedPointer<QCPGraphDataContainer>::create())
{
    ui->setupUi(this);
    ui->controls->hide();
}

PCAPlot::PCAPlot(Session &s, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PCAPlot),
    session(&s),
    data(QSharedPointer<QCPGraphDataContainer>::create())
{
    ui->setupUi(this);
    for ( size_t i = 0; i < session->gaFitter().results().size(); i++ )
        ui->fits->addItem(QString("Fit %1 (%2)").arg(i).arg(session->gaFitter().results().at(i).resultIndex, 4, 10, QChar('0')));
    connect(&session->gaFitter(), &GAFitter::done, [=](){
        ui->fits->addItem(QString("Fit %1 (%2)").arg(ui->fits->count()).arg(session->gaFitter().results().back().resultIndex, 4, 10, QChar('0')));
    });
    connect(ui->fits, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), [=](int i){
        ui->epoch->setMaximum(session->gaFitter().results().at(i).epochs);
        compute();
    });
    connect(ui->epoch, static_cast<void(QSpinBox::*)(int)>(&QSpinBox::valueChanged), this, &PCAPlot::compute);

    init(&session->project.universal());
}

PCAPlot::~PCAPlot()
{
    delete ui;
}

void PCAPlot::init(const UniversalLibrary *lib)
{
    this->lib = lib;

    QCPGraph *g = ui->plot->addGraph();
    g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssDisc, 2));
    g->setLineStyle(QCPGraph::lsNone);
    g->setData(data);
}

void PCAPlot::replot()
{
    if ( !lib )
        return;

    if ( data->size() != lib->PCA_TL_size ) {
        data->set(QVector<QCPGraphData>(lib->PCA_TL_size, {0,0}));
    }

    size_t i = 0;
    for ( QCPGraphData &p : *data ) {
        p.key = lib->PCA_TL[i];
        p.value = lib->PCA_TL[lib->PCA_TL_size + i];
        ++i;
    }
    data->sort();
    ui->plot->rescaleAxes();
    ui->plot->xAxis->setTicks(false);
    ui->plot->yAxis->setTicks(false);
    ui->plot->xAxis->grid()->setVisible(false);
    ui->plot->yAxis->grid()->setVisible(false);
    ui->plot->axisRect()->setupFullAxesBox();
    ui->plot->replot();
}

void PCAPlot::compute()
{
    if ( !session || session->busy() || ui->fits->currentIndex() < 0 )
        return;

    QFile basefile(session->resultFilePath(session->gaFitter().results().at(ui->fits->currentIndex()).resultIndex));
    PopLoader loader(basefile, session->project.universal());
    loader.load(ui->epoch->value(), session->project.universal());
    session->project.universal().pushParams();
    std::vector<scalar> singular_values = session->project.universal().get_params_principal_components(2);
    for ( const scalar &s : singular_values )
        std::cout << s << '\t';
    std::cout << std::endl;
    replot();
}

void PCAPlot::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->plot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(),
                      QString("Fit %1, epoch %2").arg(session->gaFitter().results().at(ui->fits->currentIndex()).resultIndex).arg(ui->epoch->value()));
}
