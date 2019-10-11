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


#include "samplingprofileplotter.h"
#include "ui_samplingprofileplotter.h"
#include "deckwidget.h"

SamplingProfilePlotter::SamplingProfilePlotter(Session &s, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::SamplingProfilePlotter),
    session(s),
    updating(false)
{
    ui->setupUi(this);

    connect(&session.samplingProfiler(), SIGNAL(done()), this, SLOT(updateProfiles()));

    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(setProfile(int)));
    connect(ui->x, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->y, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));

    connect(ui->hideUnselected, SIGNAL(clicked(bool)), this, SLOT(hideUnselected()));
    connect(ui->showAll, SIGNAL(clicked(bool)), this, SLOT(showAll()));

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectPlottables | QCP::iMultiSelect);
    ui->plot->setMultiSelectModifier(Qt::ControlModifier);
    connect(ui->plot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->plot->selectedAxes();
        if ( axes.isEmpty() )
           axes = ui->plot->axisRect()->axes();
        ui->plot->axisRect()->setRangeZoomAxes(axes);
        ui->plot->axisRect()->setRangeDragAxes(axes);
    });
    ui->plot->axisRect()->setRangeZoomAxes(ui->plot->axisRect()->axes());
    ui->plot->axisRect()->setRangeDragAxes(ui->plot->axisRect()->axes());

    connect(ui->plot, &QCustomPlot::mousePress, [=](QMouseEvent *e){
        ui->plot->setSelectionRectMode(e->button() == Qt::LeftButton ? QCP::srmNone : QCP::srmSelect);
    });

    connect(ui->plot, &QCustomPlot::mouseDoubleClick, [=](QMouseEvent*){
        QList<QCPAxis*> axes = ui->plot->axisRect()->rangeDragAxes(Qt::Vertical)
                             + ui->plot->axisRect()->rangeDragAxes(Qt::Horizontal);
        for ( QCPAxis *ax : axes ) {
            if ( ax->range().lower == 0 )
                ax->rescale();
            else
                ax->setRangeLower(0);
        }
        ui->plot->replot();
    });

    connect(ui->rho_normalisation, static_cast<void(QComboBox::*)(int)>(&QComboBox::currentIndexChanged), this, &SamplingProfilePlotter::updateTable);

    connect(ui->table, &QTableWidget::itemSelectionChanged, this, [=](){
        for ( DataPoint &p : points )
            p.selected = false;
        for ( QModelIndex &r : ui->table->selectionModel()->selectedRows() ) {
            size_t idx = ui->table->item(r.row(), 0)->text().toInt();
            for ( DataPoint &p : points ) {
                if ( p.idx == idx ) {
                    p.selected = true;
                    break;
                }
            }
        }
        replot(Selection::Data);
    });

    updateProfiles();
}

SamplingProfilePlotter::~SamplingProfilePlotter()
{
    delete ui;
}

void SamplingProfilePlotter::updateProfiles()
{
    for ( size_t i = ui->profile->count(); i < session.samplingProfiler().profiles().size(); i++ ) {
        const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(i);
        ui->profile->addItem(QString("Profile %1, probing %2 on %3")
                             .arg(i)
                             .arg(QString::fromStdString(session.project.model().adjustableParams[prof.target].name))
                             .arg(prof.src.prettyName()));
    }
}

void SamplingProfilePlotter::setProfile(int idx)
{
    if ( idx < 0 )
        return;

    updating = true;

    const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(idx);

    // Clear dynamic dimensions from x/y selection
    int x = ui->x->currentIndex();
    int y = ui->y->currentIndex();
    for ( int i = ui->x->count(); i >= nFixedColumns; i-- ) {
        ui->x->removeItem(i);
        ui->y->removeItem(i);
    }

    // Update dynamic dimensions
    std::vector<MAPEDimension> dim;
    if ( prof.src.archive() ) {
        dim = prof.src.archive()->searchd(session).mapeDimensions;
        ui->table->setColumnCount(nFixedColumns + dim.size());
        ui->paretoDims->setRowCount(nFixedColumns + dim.size() - 2); // paretoDims does not include Stimulation (#0) or score (#8)
        paretoGroups.resize(nFixedColumns + dim.size() - 2, nullptr);
        scoreChecks.resize(nFixedColumns + dim.size() - 2, nullptr);
        for ( size_t j = 0; j < dim.size(); j++ ) {
            QString str = QString::fromStdString(toString(dim[j].func));
            ui->table->setHorizontalHeaderItem(nFixedColumns + j, new QTableWidgetItem(str));
            ui->x->addItem(str);
            ui->y->addItem(str);
            ui->paretoDims->setVerticalHeaderItem(nFixedColumns + j - 2, new QTableWidgetItem(str));
        }
    } else {
        ui->table->setColumnCount(nFixedColumns);
        ui->paretoDims->setRowCount(nFixedColumns - 2);
        paretoGroups.resize(nFixedColumns - 2, nullptr);
        scoreChecks.resize(nFixedColumns - 2, nullptr);
    }
    ui->x->setCurrentIndex(x);
    ui->y->setCurrentIndex(y);

    // Update data points
    points.clear();
    points.resize(prof.rho_weighted.size(), DataPoint{0,0,0,false,false});
    for ( size_t i = 0; i < points.size(); i++ )
        points[i].idx = i;

    // Update pareto dimensions
    for ( size_t row = 0; row < paretoGroups.size(); row++ ) {
        if ( !paretoGroups[row] ) {
            paretoGroups[row] = new QButtonGroup(this);
            for ( int col = 0; col < 3; col++ ) {
                QRadioButton *btn = new QRadioButton();
                btn->setChecked(col == 2);
                ui->paretoDims->setCellWidget(row, col, btn);
                paretoGroups[row]->addButton(btn, col);
            }
            QCheckBox *cb = new QCheckBox();
            ui->paretoDims->setCellWidget(row, 3, cb);
            scoreChecks[row] = cb;
            connect(cb, &QCheckBox::stateChanged, this, &SamplingProfilePlotter::updateTable);
        }
    }

    // Update table
    updateTable();

    updating = false;
    replot(Selection::None);
}

double SamplingProfilePlotter::value(int i,
                                     int dimension,
                                     const SamplingProfiler::Profile &prof,
                                     const std::vector<MAPElite> &elites,
                                     const std::vector<MAPEDimension> &dim)
{
    if ( dimension == 0 )
        return i;
    else if ( dimension == 1 )
        return prof.rho_weighted[i];
    else if ( dimension == 2 )
        return prof.rho_unweighted[i];
    else if ( dimension == 3 )
        return prof.rho_target_only[i];
    else if ( dimension == 4 )
        return prof.grad_weighted[i];
    else if ( dimension == 5 )
        return prof.grad_unweighted[i];
    else if ( dimension == 6 )
        return prof.grad_target_only[i];
    else if ( dimension == 7 )
        return elites.at(i).fitness;
    else if ( dimension == 8 ) {
        double score = 1;
        for ( size_t j = 2; j < dim.size() + nFixedColumns; j++ ) {
            int jj = (j < nFixedColumns) ? j-1 : j;
            if ( scoreChecks[j-2]->isChecked() )
                score *= (value(i, jj, prof, elites, dim) - minima[j-2]) / (maxima[j-2] - minima[j-2]);
        }
        return score;
    }
    else
        return dim.at(dimension-nFixedColumns).bin_inverse(
                elites.at(i).bin[dimension-nFixedColumns],
                Wavegen::mape_multiplier(prof.src.archive()->precision));
}

void SamplingProfilePlotter::replot(Selection sel, bool showAll)
{
    if ( updating || ui->profile->currentIndex() < 0 || ui->x->currentIndex() < 0 || ui->y->currentIndex() < 0 )
        return;

    ui->plot->xAxis->setLabel(ui->x->currentText());
    ui->plot->yAxis->setLabel(ui->y->currentText());

    const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(ui->profile->currentIndex());
    std::vector<MAPElite> elites = prof.src.elites();
    std::vector<MAPEDimension> dim;
    if ( prof.src.archive() )
        dim = prof.src.archive()->searchd(session).mapeDimensions;

    // Populate data points
    std::vector<size_t> shownPoints;
    shownPoints.reserve(points.size());
    for ( size_t i = 0; i < points.size(); i++ ) {
        points[i].key = value(points[i].idx, ui->x->currentIndex(), prof, elites, dim);
        points[i].value = value(points[i].idx, ui->y->currentIndex(), prof, elites, dim);
        if ( sel != Selection::Data )
            points[i].selected = false;
        if ( !points[i].hidden )
            shownPoints.push_back(i);
    }

    // Maintain selection: Use currently plotted points to set "selected" flag on the new points
    if ( sel == Selection::Plot )
        for ( const QCPDataRange &r : ui->plot->graph()->selection().dataRanges() )
            for ( int i = r.begin(); i < r.end(); i++ )
                points[shownPoints[i]].selected = true;

    // If requested, clear hidden flag
    if ( showAll )
        for ( DataPoint &p : points )
            p.hidden = false;

    // Sort data points by new key
    std::sort(points.begin(), points.end(), [](const DataPoint &lhs, const DataPoint &rhs){ return lhs.key < rhs.key; });

    // Translate to plottable format
    QVector<double> keys, values;
    keys.reserve(points.size()), values.reserve(points.size());
    for ( DataPoint &p : points ) {
        if ( !p.hidden ) {
            keys.push_back(p.key);
            values.push_back(p.value);
        }
    }

    // Plot points
    ui->plot->clearGraphs();
    QCPGraph *g = ui->plot->addGraph();
    g->setData(keys, values, true);

    // Maintain selection: Use "selected" flag set above to reconstitute the equivalent selection in the new points
    if ( sel != Selection::None ) {
        QCPDataSelection selection;
        size_t i = 0;
        for ( DataPoint &p : points ) {
            if ( !p.hidden ) {
                if ( p.selected )
                    selection.addDataRange(QCPDataRange(i, i+1), false);
                ++i;
            }
        }
        selection.simplify();
        g->setSelection(selection);
    }

    // Apply style
    g->setLineStyle(QCPGraph::lsNone);
    g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross, 5));
    g->setSelectable(QCP::stMultipleDataRanges);
    QCPSelectionDecorator *deco = new QCPSelectionDecorator();
    QCPScatterStyle sstyle(QCPScatterStyle::ssCross, Qt::red, 6);
    sstyle.setPen(QPen(Qt::red, 2));
    deco->setScatterStyle(sstyle, QCPScatterStyle::spAll);
    g->setSelectionDecorator(deco);
    ui->plot->rescaleAxes();
    ui->plot->replot();
}

void SamplingProfilePlotter::hideUnselected()
{
    // Hide everything
    std::vector<size_t> shownPoints;
    shownPoints.reserve(points.size());
    for ( size_t i = 0; i < points.size(); i++ ) {
        if ( !points[i].hidden ) {
            shownPoints.push_back(i);
            points[i].hidden = true;
        }
    }

    // Show selected
    for ( const QCPDataRange &r : ui->plot->graph()->selection().dataRanges() )
        for ( int i = r.begin(); i < r.end(); i++ )
            points[shownPoints[i]].hidden = false;

    updateTable();

    // Replot, selecting nothing
    replot(Selection::None);
}

void SamplingProfilePlotter::showAll()
{
    replot(Selection::Plot, true);
    updateTable();
}

void SamplingProfilePlotter::updateTable()
{
    int count = 0;
    for ( const DataPoint &p : points )
        count += !p.hidden;
    ui->table->setRowCount(count);

    const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(ui->profile->currentIndex());
    std::vector<MAPElite> elites = prof.src.elites();
    std::vector<MAPEDimension> dim;
    if ( prof.src.archive() )
        dim = prof.src.archive()->searchd(session).mapeDimensions;

    // Update maxima
    maxima.assign(nFixedColumns + dim.size() - 2, -__DBL_MAX__);
    minima.assign(nFixedColumns + dim.size() - 2, 0);
    for ( size_t i = 0; i < 3; i++ )
        minima[i] = 1;
    for ( const DataPoint &p : points ) {
        if ( p.hidden )
            continue;
        for ( size_t j = 2; j < dim.size() + nFixedColumns; j++ ) {
            int jj = (j < nFixedColumns) ? j-1 : j;
            maxima[j-2] = std::max(maxima[j-2], value(p.idx, jj, prof, elites, dim));
        }

        // find true rho minima
        for ( size_t j = 2; j < 5; j++ ) {
            minima[j-2] = std::min(minima[j-2], value(p.idx, j-1, prof, elites, dim));
        }
    }

    // Set rho minima to user selection; index==1 => [min(rho), max(rho)] is already set by the above
    if ( ui->rho_normalisation->currentIndex() == 0 ) // [-1, max(rho)]
        for ( size_t i = 0; i < 3; i++ )
            minima[i] = -1;
    else if ( ui->rho_normalisation->currentIndex() == 2 ) // [min(min(rho), 0), max(rho)]
        for ( size_t i = 0; i < 3; i++ )
            minima[i] = std::min(minima[i], 0.);
    else if ( ui->rho_normalisation->currentIndex() == 3 ) // [floor(rho), max(rho)]
        for ( size_t i = 0; i < 3; i++ )
            minima[i] = floor(minima[i]);

    // Update table contents
    ui->table->setSortingEnabled(false);
    size_t row = 0;
    for ( size_t i = 0; i < points.size(); i++ ) {
        if ( points[i].hidden )
            continue;
        for ( size_t j = 0; j < dim.size() + nFixedColumns; j++ ) {
            QTableWidgetItem *item = new QTableWidgetItem();
            item->setFlags(Qt::ItemIsEnabled | Qt::ItemIsSelectable);
            item->setData(Qt::DisplayRole, value(points[i].idx, j, prof, elites, dim));
            ui->table->setItem(row, j, item);
        }
        ++row;
    }
    ui->table->setSortingEnabled(true);
    ui->table->scrollToTop();
}

void SamplingProfilePlotter::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->plot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), ui->profile->currentText());
}

int dominates(const std::vector<double> &lhs, const std::vector<double> &rhs, const std::vector<int> &direction)
{
    int dir = 0, strictDir = 0, dom = 0;
    for ( size_t i = 0; i < lhs.size(); i++ ) {
        if ( lhs[i] > rhs[i] )
            dir = direction[i];
        else if ( lhs[i] < rhs[i] )
            dir = -direction[i];
        else
            continue;
        if ( dir ) {
            if ( strictDir && dir != strictDir )
                return 0;
            strictDir = dir;
            dom += dir;
        }
    }
    return dom;
}

void SamplingProfilePlotter::on_pareto_clicked()
{
    std::vector<int> dims;
    std::vector<int> direction;
    for ( size_t i = 0; i < paretoGroups.size(); i++ ) {
        if ( paretoGroups[i]->checkedId() == 2 )
            continue;
        dims.push_back(i+1); // +1 to compensate for lack of stimIdx in pareto table
        direction.push_back(2*paretoGroups[i]->checkedId() - 1);
    }

    int nPoints = 0, nDims = dims.size();
    for ( const DataPoint &p : points )
        nPoints += !p.hidden;

    const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(ui->profile->currentIndex());
    std::vector<MAPElite> elites = prof.src.elites();
    std::vector<MAPEDimension> mapedim;
    if ( prof.src.archive() )
        mapedim = prof.src.archive()->searchd(session).mapeDimensions;

    std::vector<std::vector<double>> myPoints(nPoints, std::vector<double>(nDims));
    std::vector<int> pointsIdx(nPoints);
    int idx = 0;
    for ( size_t i = 0; i < points.size(); i++ ) {
        points[i].selected = false;
        if ( points[i].hidden )
            continue;
        for ( int j = 0; j < nDims; j++ ) {
            myPoints[idx][j] = value(points[i].idx, dims[j], prof, elites, mapedim);
        }
        pointsIdx[idx] = i;
        ++idx;
    }

    for ( int idx = 0; idx < nPoints; idx++ ) {
        bool dominated = false;
        for ( int ref = 0; ref < idx; ref++ ) {
            if ( !points[pointsIdx[ref]].selected )
                continue;
            int dom = dominates(myPoints[idx], myPoints[ref], direction);
            if ( dom > 0 )
                points[pointsIdx[ref]].selected = false; // idx dominates ref => deselect ref
            else if ( dom < 0 ) {
                dominated = true; // ref dominates idx => cancel search, don't select idx
                break;
            }
        }
        if ( !dominated )
            points[pointsIdx[idx]].selected = true;
    }

    replot(Selection::Data);
}

void SamplingProfilePlotter::on_addDeckGo_clicked()
{
    if ( ui->table->selectedItems().empty() )
        return;
    int stimIdx = ui->table->selectedItems().first()->data(Qt::DisplayRole).toInt();

    if ( ui->profile->currentIndex() < 0 )
        return;
    const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(ui->profile->currentIndex());

    DeckWidget *dw;
    if ( ui->addDeckCombo->currentIndex() < 1 ) {
        dw = new DeckWidget(session, this);
        dw->clear();
        int idx = ui->tabWidget->count() - 3;
        QString label = QString("[%1]").arg(idx + session.wavesets().decks().size());
        ui->tabWidget->addTab(dw, label);
        ui->addDeckCombo->addItem(label);
        ui->addDeckCombo->setCurrentIndex(idx + 1);
    } else {
        dw = dynamic_cast<DeckWidget*>(ui->tabWidget->widget(ui->addDeckCombo->currentIndex() + 2));
        if ( !dw )
            return;
    }

    dw->select(prof.target, prof.src, stimIdx);

    ui->tabWidget->setCurrentWidget(dw);
}
