#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "config.h"
#include "project.h"

WavegenDialog::WavegenDialog(Session *s, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    session(s),
    wavegen(s->wavegen()),
    abort(false)
{
    ui->setupUi(this);
    this->setWindowFlags(Qt::Window);
    initWG();
    initPlotControls();
}

void WavegenDialog::initWG()
{
    ui->btnPermute->setEnabled(session->project.wgPermute());
    for ( const AdjustableParam &p : wavegen.lib.model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(this, SIGNAL(permute()), &wavegen, SLOT(permute()));
    connect(this, SIGNAL(adjustSigmas()), &wavegen, SLOT(adjustSigmas()));
    connect(this, SIGNAL(search(int)), &wavegen, SLOT(search(int)));
    connect(&wavegen, SIGNAL(startedSearch(int)), this, SLOT(startedSearch(int)));
    connect(&wavegen, SIGNAL(searchTick(int)), this, SLOT(searchTick(int)));
    connect(&wavegen, SIGNAL(done(int)), this, SLOT(end(int)));

    connect(ui->btnPermute, &QPushButton::clicked, [&](bool){
        ui->log->addItem("Parameter permutation begins...");
        ui->log->scrollToBottom();
        actions.push_back("Parameter permutation");
        emit permute();
    });
    connect(ui->btnSigadjust, &QPushButton::clicked, [&](bool){
        ui->log->addItem("Sigma adjustment begins...");
        ui->log->scrollToBottom();
        actions.push_back("Sigma adjustment");
        emit adjustSigmas();
    });
    connect(ui->btnSearchAll, &QPushButton::clicked, [&](bool){
        for ( size_t i = 0; i < wavegen.lib.model.adjustableParams.size(); i++ ) {
            actions.push_back(QString("Search for %1").arg(QString::fromStdString(wavegen.lib.model.adjustableParams[i].name)));
            emit search(i);
        }
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        int i = ui->cbSearch->currentIndex();
        actions.push_back(QString("Search for %1").arg(QString::fromStdString(wavegen.lib.model.adjustableParams[i].name)));
        emit search(i);
    });

    connect(ui->btnAbort, &QPushButton::clicked, [&](bool){
        wavegen.abort();
        abort = true;
    });

    session->setWavegenData(Config::Wavegen);
    session->setStimulationData(Config::Stimulation);
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
}

void WavegenDialog::startedSearch(int param)
{
    ui->log->addItem(QString("%1 begins...").arg(actions.front()));
    ui->log->addItem("");
    ui->log->scrollToBottom();
    refreshPlotControls();
}

void WavegenDialog::searchTick(int i)
{
    ui->log->item(ui->log->count()-1)->setText(QString("%1 iterations...").arg(i));
}

void WavegenDialog::end(int arg)
{
    QString outcome = abort ? "aborted" : "complete";
    ui->log->addItem(QString("%1 %2.").arg(actions.front(), outcome));
    ui->log->scrollToBottom();
    if ( abort )
        actions.clear();
    else
        actions.pop_front();
    abort = false;
    refreshPlotControls();
}

void WavegenDialog::initPlotControls()
{
    int i = 0, n = wavegen.searchd.mapeDimensions.size();
    groupx = new QButtonGroup(this);
    groupy = new QButtonGroup(this);
    mins.resize(n);
    maxes.resize(n);
    ui->plotTable->setRowCount(n);
    ui->plotTable->setColumnWidth(0, 25);
    ui->plotTable->setColumnWidth(1, 25);
    QStringList labels;
    for ( MAPEDimension const& d : wavegen.searchd.mapeDimensions ) {
        labels << QString::fromStdString(toString(d.func));

        QRadioButton *x = new QRadioButton();
        groupx->addButton(x, i);
        ui->plotTable->setCellWidget(i, 0, x);

        QRadioButton *y = new QRadioButton();
        groupy->addButton(y, i);
        ui->plotTable->setCellWidget(i, 1, y);

        QDoubleSpinBox *min = new QDoubleSpinBox();
        min->setRange(d.min, d.max);
        min->setValue(d.min);
        ui->plotTable->setCellWidget(i, 2, min);
        mins[i] = min;

        QDoubleSpinBox *max = new QDoubleSpinBox();
        max->setRange(d.min, d.max);
        max->setValue(d.max);
        ui->plotTable->setCellWidget(i, 3, max);
        maxes[i] = max;

        ++i;
    }
    ui->plotTable->setVerticalHeaderLabels(labels);

    for ( AdjustableParam const& p : wavegen.lib.model.adjustableParams )
        ui->cbPlot->addItem(QString::fromStdString(p.name));

    connect(ui->btnPlotApply, SIGNAL(clicked(bool)), this, SLOT(replot()));
    connect(ui->cbPlot, SIGNAL(currentIndexChanged(int)), this, SLOT(setPlotMinMaxSteps(int)));

    // Plot setup
    ui->plot->setInteractions(QCP::iRangeDrag|QCP::iRangeZoom); // this will also allow rescaling the color scale by dragging/zooming
    ui->plot->axisRect()->setupFullAxesBox(true);

    colorMap = new QCPColorMap(ui->plot->xAxis, ui->plot->yAxis);
    colorMap->setInterpolate(false);

    // add a color scale:
    QCPColorScale *colorScale = new QCPColorScale(ui->plot);
    ui->plot->plotLayout()->addElement(0, 1, colorScale); // add it to the right of the main axis rect
    colorScale->setType(QCPAxis::atRight); // scale shall be vertical bar with tick/axis labels right (actually atRight is already the default)
    colorMap->setColorScale(colorScale); // associate the color map with the color scale
    colorScale->axis()->setLabel("Fitness");

    // Make a heat gradient with blue for exactly 0, to disambiguate between "no data" and "really low fitness"
    QCPColorGradient foo(QCPColorGradient::gpHot);
    foo.setColorStopAt(0, QColor("blue"));
    foo.setColorStopAt(__DBL_MIN__, QColor("black"));
    colorMap->setGradient(foo);

    // make sure the axis rect and color scale synchronize their bottom and top margins (so they line up):
    QCPMarginGroup *marginGroup = new QCPMarginGroup(ui->plot);
    ui->plot->axisRect()->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);
    colorScale->setMarginGroup(QCP::msBottom|QCP::msTop, marginGroup);

    refreshPlotControls();
}

void WavegenDialog::refreshPlotControls()
{
    int i = 0;
    bool enabled = false;
    for ( std::list<MAPElite> const& l : wavegen.completedArchives ) {
        QStandardItem *item = qobject_cast<const QStandardItemModel*>(ui->cbPlot->model())->item(i++);
        item->setFlags(l.empty()
                       ? item->flags() & ~(Qt::ItemIsSelectable|Qt::ItemIsEnabled)
                       : item->flags() | (Qt::ItemIsSelectable|Qt::ItemIsEnabled));
        if ( !l.empty() )
            enabled = true;
    }
    ui->btnPlotApply->setEnabled(enabled);
    ui->btnAddToSel->setEnabled(enabled);

    ui->cbSelections->clear();
    ui->cbSelections->addItem("Selection...");
    for ( Selection const& sel : selections ) {
        ui->cbSelections->addItem(name(sel));
    }

    setPlotMinMaxSteps(ui->cbPlot->currentIndex());
}

void WavegenDialog::setPlotMinMaxSteps(int p)
{
    int mult = wavegen.mape_multiplier(wavegen.archivePrecision.at(p));
    int i = 0;
    for ( MAPEDimension const& d : wavegen.searchd.mapeDimensions ) {
        double step = (d.max - d.min) / (mult * d.resolution);
        mins[i]->setSingleStep(step);
        mins[i]->setMaximum(d.max-step);
        maxes[i]->setSingleStep(step);
        ++i;
    }
}

// Populate currentSelection from plot box settings in the GUI
bool WavegenDialog::select()
{
    Selection sel;
    sel.cx = groupx->checkedId(); // The selection for the plot's X axis
    sel.cy = groupy->checkedId(); // The selection for the plot's Y axis
    sel.param = ui->cbPlot->currentIndex(); // The selected parameter or waveform archive

    if ( sel.cx == sel.cy || sel.cx < 0 || sel.cy < 0 || sel.param < 0 || wavegen.completedArchives[sel.param].empty() )
        return false;

    size_t resolution_multiplier = wavegen.mape_multiplier(wavegen.archivePrecision.at(sel.param));

    size_t nDimensions = wavegen.searchd.mapeDimensions.size();
    std::vector<std::pair<size_t, size_t>> ranges(nDimensions);
    sel.min.resize(nDimensions);
    sel.max.resize(nDimensions);
    for ( size_t i = 0; i < nDimensions; i++ ) {
        sel.min[i] = mins[i]->value();
        sel.max[i] = maxes[i]->value();
        MAPEDimension const& d(wavegen.searchd.mapeDimensions.at(i));
        ranges[i].first = resolution_multiplier * d.resolution * (sel.min[i]-d.min)/(d.max-d.min);
        ranges[i].second = resolution_multiplier * d.resolution * (sel.max[i]-d.min)/(d.max-d.min);
        if ( ranges[i].second <= ranges[i].first )
            ranges[i].second = ranges[i].first + 1;
    }

    sel.nx = ranges[sel.cx].second - ranges[sel.cx].first;
    sel.ny = ranges[sel.cy].second - ranges[sel.cy].first;
    sel.elites = std::vector<MAPElite>(sel.nx * sel.ny);

    for ( MAPElite const& e : wavegen.completedArchives.at(sel.param) ) {
        bool in_range(true);
        for ( size_t j = 0; j < nDimensions; j++ ) {
            if ( e.bin[j] < ranges[j].first || e.bin[j] >= ranges[j].second ) {
                in_range = false;
                break;
            }
        }
        if ( in_range ) {
            // set ix, iy as indices to sel.elites, relative to selection boundaries:
            size_t ix = e.bin[sel.cx] - ranges[sel.cx].first, iy = e.bin[sel.cy] - ranges[sel.cy].first;
            if ( sel.elites[ix + sel.nx*iy].compete(e) )
                sel.elites[ix + sel.nx*iy].bin = e.bin;
        }
    }

    // Ensure min and max element have valid bin values (to transmit ranges[cx/cy])
    if ( sel.elites.front().bin.size() == 0 ) {
        sel.elites.front().bin.resize(nDimensions);
        sel.elites.front().bin[sel.cx] = ranges[sel.cx].first;
        sel.elites.front().bin[sel.cy] = ranges[sel.cy].first;
    }
    if ( sel.elites.back().bin.size() == 0 ) {
        sel.elites.back().bin.resize(nDimensions);
        sel.elites.back().bin[sel.cx] = ranges[sel.cx].second;
        sel.elites.back().bin[sel.cy] = ranges[sel.cy].second;
    }

    currentSelection = std::move(sel);
    ui->cbSelections->setCurrentIndex(0); // Clear current selection in GUI upon creating a new one

    return true;
}

void WavegenDialog::replot(bool doSelect)
{
    if ( doSelect && !select() )
        return;

    MAPEDimension const& dimx = wavegen.searchd.mapeDimensions[currentSelection.cx];
    MAPEDimension const& dimy = wavegen.searchd.mapeDimensions[currentSelection.cy];

    size_t resolution_multiplier = wavegen.mape_multiplier(wavegen.archivePrecision.at(currentSelection.param));
    int nx = dimx.resolution * resolution_multiplier;
    int ny = dimy.resolution * resolution_multiplier;

    // Set up axes
    ui->plot->xAxis->setLabel(QString::fromStdString(toString(dimx.func)));
    ui->plot->yAxis->setLabel(QString::fromStdString(toString(dimy.func)));

    // set up the QCPColorMap:
    colorMap->data()->clear();
    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(dimx.min, dimx.max), QCPRange(dimy.min, dimy.max));
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:
    // Note, plot area spans the full dimensional range, but only selection is assigned
    for ( int ix = 0; ix < currentSelection.nx; ++ix )
        for ( int iy = 0; iy < currentSelection.ny; ++iy )
            colorMap->data()->setCell(
                    ix + currentSelection.elites.front().bin[currentSelection.cx],
                    iy + currentSelection.elites.front().bin[currentSelection.cy],
                    currentSelection.elites[ix + currentSelection.nx*iy].stats.fitness);

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    ui->plot->rescaleAxes();
    ui->plot->replot();

}

void WavegenDialog::on_btnAddToSel_clicked()
{
    if ( select() ) {
        selections.push_back(std::move(currentSelection));
        refreshPlotControls();
        ui->cbSelections->setCurrentIndex(0);
        replot();
        emit selectionsChanged();
    }
}

void WavegenDialog::on_cbSelections_currentIndexChanged(int index)
{
    if ( index <= 0 )
        return;
    Selection const& sel = selections.at(index-1);
    ui->cbPlot->setCurrentIndex(sel.param);
    for ( size_t i = 0; i < wavegen.searchd.mapeDimensions.size(); i++ ) {
        mins[i]->setValue(sel.min[i]);
        maxes[i]->setValue(sel.max[i]);
        groupx->button(i)->setChecked(i == sel.cx);
        groupy->button(i)->setChecked(i == sel.cy);
    }
    currentSelection = sel;
    replot(false);
}

QString WavegenDialog::name(const Selection &sel) const
{
    return QString("%1: %2:%3 (%4x%5 bins)")
            .arg(QString::fromStdString(wavegen.lib.model.adjustableParams[sel.param].name))
            .arg(QString::fromStdString(toString(wavegen.searchd.mapeDimensions[sel.cx].func)))
            .arg(QString::fromStdString(toString(wavegen.searchd.mapeDimensions[sel.cy].func)))
            .arg(sel.nx).arg(sel.ny);
}
