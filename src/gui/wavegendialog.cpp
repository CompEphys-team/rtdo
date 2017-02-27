#include "wavegendialog.h"
#include "ui_wavegendialog.h"
#include "config.h"

WavegenDialog::WavegenDialog(MetaModel &model, QThread *thread, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::WavegenDialog),
    thread(thread),
    model(model),
    lib(model, Config::WavegenLibrary, Config::Run),
    wg(new Wavegen(lib, Config::Stimulation, Config::Wavegen)),
    abort(false)
{
    ui->setupUi(this);
    initWG();
    initPlotControls();
}

void WavegenDialog::initWG()
{
    ui->btnPermute->setEnabled(lib.compileD.permute);
    for ( const AdjustableParam &p : model.adjustableParams )
        ui->cbSearch->addItem(QString::fromStdString(p.name));

    connect(this, SIGNAL(permute()), wg, SLOT(permute()));
    connect(this, SIGNAL(adjustSigmas()), wg, SLOT(adjustSigmas()));
    connect(this, SIGNAL(search(int)), wg, SLOT(search(int)));
    connect(wg, SIGNAL(startedSearch(int)), this, SLOT(startedSearch(int)));
    connect(wg, SIGNAL(searchTick(int)), this, SLOT(searchTick(int)));
    connect(wg, SIGNAL(done(int)), this, SLOT(end(int)));

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
        for ( size_t i = 0; i < model.adjustableParams.size(); i++ ) {
            actions.push_back(QString("Search for %1").arg(QString::fromStdString(model.adjustableParams[i].name)));
            emit search(i);
        }
    });
    connect(ui->btnSearchOne, &QPushButton::clicked, [&](bool){
        int i = ui->cbSearch->currentIndex();
        actions.push_back(QString("Search for %1").arg(QString::fromStdString(model.adjustableParams[i].name)));
        emit search(i);
    });

    connect(ui->btnAbort, &QPushButton::clicked, [&](bool){
        wg->abort();
        abort = true;
    });

    wg->moveToThread(thread);
}

WavegenDialog::~WavegenDialog()
{
    delete ui;
    delete wg;
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
    int i = 0;
    groupx = new QButtonGroup(this);
    groupy = new QButtonGroup(this);
    ui->plotTable->setRowCount(wg->searchd.mapeDimensions.size());
    ui->plotTable->setColumnWidth(0, 25);
    ui->plotTable->setColumnWidth(1, 25);
    QStringList labels;
    for ( MAPEDimension const& d : wg->searchd.mapeDimensions ) {
        labels << QString::fromStdString(toString(d.func));

        QRadioButton *x = new QRadioButton();
        groupx->addButton(x, i);
        ui->plotTable->setCellWidget(i, 0, x);

        QRadioButton *y = new QRadioButton();
        groupy->addButton(y, i);
        ui->plotTable->setCellWidget(i, 1, y);

        ++i;
    }
    ui->plotTable->setVerticalHeaderLabels(labels);

    connect(ui->btnPlotApply, SIGNAL(clicked(bool)), this, SLOT(replot()));

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
}

void WavegenDialog::refreshPlotControls()
{
    ui->cbPlot->clear();
    int i = 0;
    for ( std::list<MAPElite> const& l : wg->completedArchives ) {
        if ( !l.empty() )
            ui->cbPlot->addItem(QString::fromStdString(model.adjustableParams[i].name));
        ++i;
    }
    ui->btnPlotApply->setEnabled(ui->cbPlot->count() > 0);
}

void WavegenDialog::replot()
{
    if ( groupx->checkedId() == groupy->checkedId()
         || groupx->checkedId() < 0
         || groupy->checkedId() < 0
         || ui->cbPlot->currentIndex() < 0
         || wg->completedArchives[ui->cbPlot->currentIndex()].empty() )
        return;

    const std::list<MAPElite> &archive = wg->completedArchives.at(ui->cbPlot->currentIndex());
    size_t resolution_multiplier = wg->mape_multiplier(wg->archivePrecision.at(ui->cbPlot->currentIndex()));

    size_t cx = groupx->checkedId();
    size_t cy = groupy->checkedId();

    MAPEDimension const& dimx = wg->searchd.mapeDimensions[cx];
    MAPEDimension const& dimy = wg->searchd.mapeDimensions[cy];

    int nx = dimx.resolution * resolution_multiplier;
    int ny = dimy.resolution * resolution_multiplier;
    std::vector<MAPElite> map(nx*ny);

    for ( MAPElite const& e : archive ) {
        size_t ix = e.bin[cx], iy = e.bin[cy];
        if ( map[ix + nx*iy].compete(e) )
            map[ix + nx*iy].bin = e.bin;
    }

    // Set up axes
    ui->plot->xAxis->setLabel(QString::fromStdString(toString(dimx.func)));
    ui->plot->yAxis->setLabel(QString::fromStdString(toString(dimy.func)));

    // set up the QCPColorMap:
    colorMap->data()->setSize(nx, ny); // we want the color map to have nx * ny data points
    colorMap->data()->setRange(QCPRange(dimx.min, dimx.max), QCPRange(dimx.min, dimx.max));
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:
    for (int ix=0; ix<nx; ++ix)
        for (int iy=0; iy<ny; ++iy)
            colorMap->data()->setCell(ix, iy, map[ix + nx*iy].stats.fitness);

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    ui->plot->rescaleAxes();
    ui->plot->replot();
}
