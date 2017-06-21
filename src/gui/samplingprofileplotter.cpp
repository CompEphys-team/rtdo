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
        dim = prof.src.archive()->searchd.mapeDimensions;
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

    std::vector<MAPElite> elites = prof.src.elites();
    std::vector<std::function<double(int)>> funcs;
    for ( size_t i = 0; i < nFixedColumns + dim.size(); i++ )
        funcs.push_back(valueFunction(i, prof, elites, dim));

    // Update table contents
    ui->table->setRowCount(0);
    ui->table->setSortingEnabled(false);
    ui->table->setRowCount(elites.size());
    for ( size_t i = 0; i < elites.size(); i++ ) {
        for ( size_t j = 0; j < funcs.size(); j++ ) {
            QTableWidgetItem *item = new QTableWidgetItem();
            item->setFlags(Qt::ItemIsEnabled);
            item->setData(Qt::DisplayRole, funcs[j](i));
            ui->table->setItem(i, j, item);
        }
    }
    ui->table->setSortingEnabled(true);

    updating = false;
    replot(true);
}

std::function<double(int)> SamplingProfilePlotter::valueFunction(int dimension,
                                                                 const SamplingProfiler::Profile &prof,
                                                                 const std::vector<MAPElite> &elites,
                                                                 const std::vector<MAPEDimension> &dim)
{
    const auto pprof = &prof;
    const auto pelites = &elites;
    const auto pdim = &dim;
    std::function<double(int)> fn;
    if ( dimension == 0 )
        fn = [](int i){ return i; };
    else if ( dimension == 1 )
        fn = [=](int i){ return pprof->accuracy[i]; };
    else if ( dimension == 2 )
        fn = [=](int i){ return pprof->gradient[i]; };
    else if ( dimension == 3 )
        fn = [=](int i){ return pelites->at(i).stats.fitness; };
    else
        fn = [=](int i){
            return pdim->at(dimension-nFixedColumns).bin_inverse(
                    pelites->at(i).bin[dimension-nFixedColumns],
                    Wavegen::mape_multiplier(pprof->src.archive()->precision));
        };
    return fn;
}

void SamplingProfilePlotter::replot(bool discardSelection)
{
    if ( updating || ui->profile->currentIndex() < 0 || ui->x->currentIndex() < 0 || ui->y->currentIndex() < 0 )
        return;

    const SamplingProfiler::Profile &prof = session.samplingProfiler().profiles().at(ui->profile->currentIndex());
    std::vector<MAPElite> elites = prof.src.elites();
    std::vector<MAPEDimension> dim;
    if ( prof.src.archive() )
        dim = prof.src.archive()->searchd.mapeDimensions;

    // Populate points vector from unsorted table
    std::vector<DataPoint> points;
    points.reserve(elites.size());
    std::vector<size_t> translate(elites.size());
    std::function<double(int)> xfunc = valueFunction(ui->x->currentIndex(), prof, elites, dim);
    std::function<double(int)> yfunc = valueFunction(ui->y->currentIndex(), prof, elites, dim);
    ui->table->setSortingEnabled(false);
    for ( size_t i = 0, j = 0; i < elites.size(); i++ ) {
        if ( !ui->table->isRowHidden(i) ) {
            points.push_back({xfunc(i), yfunc(i), i, false});
            translate[i] = j++;
        }
    }
    ui->table->setSortingEnabled(true);

    // Maintain selection: Use currently plotted points to set "selected" flag on the new points
    if ( !discardSelection )
        for ( const QCPDataRange &r : ui->plot->graph()->selection().dataRanges() )
            for ( int i = r.begin(); i < r.end(); i++ )
                points[translate[plottedPoints[i].idx]].selected = true;

    // Sort points by key
    std::sort(points.begin(), points.end(), [](const DataPoint &lhs, const DataPoint &rhs){ return lhs.key < rhs.key; });
    QVector<double> keys(points.size()), values(points.size());
    for ( size_t i = 0; i < points.size(); i++ ) {
        keys[i] = points[i].key;
        values[i] = points[i].value;
    }

    // Plot scatter
    ui->plot->clearGraphs();
    QCPGraph *g = ui->plot->addGraph();
    g->setData(keys, values, true);
    g->setLineStyle(QCPGraph::lsNone);
    g->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCross, 5));
    g->setSelectable(QCP::stMultipleDataRanges);
    ui->plot->rescaleAxes();

    // Maintain selection: Use flag set above to reconstitute the equivalent selection in the new points
    if ( !discardSelection ) {
        QCPDataSelection selection;
        for ( size_t i = 0; i < points.size(); i++ )
            if ( points[i].selected )
                selection.addDataRange(QCPDataRange(i, i+1), false);
        selection.simplify();
        g->setSelection(selection);
    }

    plottedPoints = std::move(points);

    ui->plot->replot();
}

void SamplingProfilePlotter::hideUnselected()
{
    replot(false); // Update plottedPoints

    ui->table->setSortingEnabled(false);
    std::vector<size_t> previouslySelected;
    previouslySelected.reserve(ui->table->rowCount());
    for ( int i = 0; i < ui->table->rowCount(); i++ ) {
        if ( !ui->table->isRowHidden(i) ) {
            ui->table->setRowHidden(i, true);
            previouslySelected.push_back(i);
        }
    }
    for ( const DataPoint &p : plottedPoints ) {
        if ( p.selected ) {
            ui->table->setRowHidden(p.idx, false);
        }
    }
    ui->table->setSortingEnabled(true);
    replot(true);
}

void SamplingProfilePlotter::showAll()
{
    for ( int i = 0; i < ui->table->rowCount(); i++ )
        ui->table->setRowHidden(i, false);
    replot();
}
