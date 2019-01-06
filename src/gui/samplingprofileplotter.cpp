#include "samplingprofileplotter.h"
#include "ui_samplingprofileplotter.h"

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
    connect(ui->fitness, SIGNAL(toggled(bool)), this, SLOT(replot()));
    connect(ui->gradient, SIGNAL(toggled(bool)), this, SLOT(replot()));
    connect(ui->accuracy, SIGNAL(toggled(bool)), this, SLOT(replot()));
    connect(ui->fitness, SIGNAL(toggled(bool)), this, SLOT(updateTable()));
    connect(ui->gradient, SIGNAL(toggled(bool)), this, SLOT(updateTable()));
    connect(ui->accuracy, SIGNAL(toggled(bool)), this, SLOT(updateTable()));

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
        for ( size_t j = 0; j < dim.size(); j++ ) {
            QString str = QString::fromStdString(toString(dim[j].func));
            ui->table->setHorizontalHeaderItem(nFixedColumns + j, new QTableWidgetItem(str));
            ui->x->addItem(str);
            ui->y->addItem(str);
        }
    }
    ui->x->setCurrentIndex(x);
    ui->y->setCurrentIndex(y);

    // Update data points
    points.clear();
    points.resize(prof.rho_weighted.size(), DataPoint{0,0,0,false,false});
    for ( size_t i = 0; i < points.size(); i++ )
        points[i].idx = i;

    // Update table
    updateTable();

    updating = false;
    replot(true);
}

double SamplingProfilePlotter::value(int i,
                                     int dimension,
                                     const SamplingProfiler::Profile &prof,
                                     const std::vector<MAPElite> &elites,
                                     const std::vector<MAPEDimension> &dim,
                                     const ScoreStruct &sstr)
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
        return elites.at(i).fitness;
    else if ( dimension == 5 )
        return 0;
//        return ( (sstr.weightF * (elites.at(i).fitness-sstr.minF))
//               + (sstr.weightG * (prof.gradient[i]-sstr.minG))
//               + (sstr.weightA * (prof.accuracy[i]-sstr.minA))
//               ) / sstr.norm;
    else if ( dimension == 6 )
        return 0;
//        return ( (sstr.sweightF * (elites.at(i).fitness-sstr.sminF))
//               + (sstr.sweightG * (prof.gradient[i]-sstr.sminG))
//               + (sstr.sweightA * (prof.accuracy[i]-sstr.sminA))
//               ) / sstr.snorm;
    else
        return dim.at(dimension-nFixedColumns).bin_inverse(
                elites.at(i).bin[dimension-nFixedColumns],
                Wavegen::mape_multiplier(prof.src.archive()->precision));
}

SamplingProfilePlotter::ScoreStruct SamplingProfilePlotter::getScoreStruct(
        const SamplingProfiler::Profile &prof,
        const std::vector<MAPElite> &elites,
        bool scoreF, bool scoreG, bool scoreA)
{
    ScoreStruct ret;
//    for ( size_t i = 0; i < points.size(); i++ ) {
//        if ( elites.at(points[i].idx).fitness < ret.minF )  ret.minF = elites.at(points[i].idx).fitness;
//        if ( elites.at(points[i].idx).fitness > ret.maxF )  ret.maxF = elites.at(points[i].idx).fitness;
//        if ( prof.accuracy[points[i].idx] < ret.minA )      ret.minA = prof.accuracy[points[i].idx];
//        if ( prof.accuracy[points[i].idx] > ret.maxA )      ret.maxA = prof.accuracy[points[i].idx];
//        if ( prof.gradient[points[i].idx] < ret.minG )      ret.minG = prof.gradient[points[i].idx];
//        if ( prof.gradient[points[i].idx] > ret.maxG )      ret.maxG = prof.gradient[points[i].idx];
//        if ( points[i].hidden )
//            continue;
//        if ( elites.at(points[i].idx).fitness < ret.sminF )  ret.sminF = elites.at(points[i].idx).fitness;
//        if ( elites.at(points[i].idx).fitness > ret.smaxF )  ret.smaxF = elites.at(points[i].idx).fitness;
//        if ( prof.accuracy[points[i].idx] < ret.sminA )      ret.sminA = prof.accuracy[points[i].idx];
//        if ( prof.accuracy[points[i].idx] > ret.smaxA )      ret.smaxA = prof.accuracy[points[i].idx];
//        if ( prof.gradient[points[i].idx] < ret.sminG )      ret.sminG = prof.gradient[points[i].idx];
//        if ( prof.gradient[points[i].idx] > ret.smaxG )      ret.smaxG = prof.gradient[points[i].idx];
//    }
//    // score = sum(weight * (value-min)/(max-min)) / sum(weights) | weight = (max-min)/(max+min)
//    ret.weightF = scoreF/(ret.maxF+ret.minF);
//    ret.weightG = scoreG/(ret.maxG+ret.minG);
//    ret.weightA = scoreA/(ret.maxA+ret.minA);
//    ret.norm = (ret.maxF-ret.minF)*ret.weightF + (ret.maxG-ret.minG)*ret.weightG + (ret.maxA-ret.minA)*ret.weightA;
//    ret.sweightF = scoreF/(ret.smaxF+ret.sminF);
//    ret.sweightG = scoreG/(ret.smaxG+ret.sminG);
//    ret.sweightA = scoreA/(ret.smaxA+ret.sminA);
//    ret.snorm = (ret.smaxF-ret.sminF)*ret.sweightF + (ret.smaxG-ret.sminG)*ret.sweightG + (ret.smaxA-ret.sminA)*ret.sweightA;
    return ret;
}

void SamplingProfilePlotter::replot(bool discardSelection, bool showAll)
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
    ScoreStruct sstr = getScoreStruct(prof, elites, ui->fitness->isChecked(), ui->gradient->isChecked(), ui->accuracy->isChecked());

    // Populate data points
    std::vector<size_t> shownPoints;
    shownPoints.reserve(points.size());
    for ( size_t i = 0; i < points.size(); i++ ) {
        points[i].key = value(points[i].idx, ui->x->currentIndex(), prof, elites, dim, sstr);
        points[i].value = value(points[i].idx, ui->y->currentIndex(), prof, elites, dim, sstr);
        points[i].selected = false;
        if ( !points[i].hidden )
            shownPoints.push_back(i);
    }

    // Maintain selection: Use currently plotted points to set "selected" flag on the new points
    if ( !discardSelection )
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
    if ( !discardSelection ) {
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
    replot(true);
}

void SamplingProfilePlotter::showAll()
{
    replot(false, true);
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
    ScoreStruct sstr = getScoreStruct(prof, elites, ui->fitness->isChecked(), ui->gradient->isChecked(), ui->accuracy->isChecked());

    // Update table contents
    ui->table->setSortingEnabled(false);
    size_t row = 0;
    for ( size_t i = 0; i < points.size(); i++ ) {
        if ( points[i].hidden )
            continue;
        for ( size_t j = 0; j < dim.size() + nFixedColumns; j++ ) {
            QTableWidgetItem *item = new QTableWidgetItem();
            item->setFlags(Qt::ItemIsEnabled);
            item->setData(Qt::DisplayRole, value(points[i].idx, j, prof, elites, dim, sstr));
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
