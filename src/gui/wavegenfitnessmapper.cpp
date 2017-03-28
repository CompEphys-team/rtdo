#include "wavegenfitnessmapper.h"
#include "ui_wavegenfitnessmapper.h"
#include <QButtonGroup>
#include <QDoubleSpinBox>

WavegenFitnessMapper::WavegenFitnessMapper(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::WavegenFitnessMapper),
    session(session),
    savedSelection(false)
{
    ui->setupUi(this);
    initPlot();

    ui->dimensions->setColumnWidth(0, 25);
    ui->dimensions->setColumnWidth(1, 25);

    updateCombo();
    updateDimensions();

    connect(&session, SIGNAL(actionLogged(QString,QString,QString,int)), this, SLOT(updateCombo()));
    connect(ui->combo, SIGNAL(currentIndexChanged(int)), this, SLOT(updateDimensions()));
    connect(ui->btnDraw, SIGNAL(clicked(bool)), this, SLOT(replot()));
}

WavegenFitnessMapper::~WavegenFitnessMapper()
{
    delete ui;
    delete colorMap;
}

void WavegenFitnessMapper::initPlot()
{
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

void WavegenFitnessMapper::updateCombo()
{
    int currentIdx = ui->combo->currentIndex();
    QSize currentData = ui->combo->currentData().toSize();
    ui->combo->clear();
    int i = 0;
    for ( Wavegen::Archive const& arch : session.wavegen().archives() ) {
        ui->combo->addItem(QString("%1 complete archive %2")
                           .arg(QString::fromStdString(session.project.model().adjustableParams[arch.param].name))
                           .arg(i),
                           QSize(0, i));
        ++i;
    }
    i = 0;
    for ( WavegenSelection const& sel : session.wavegenselector().selections() ) {
        ui->combo->addItem(QString("Subset %1 (%2 %3, %4 bins)")
                           .arg(i)
                           .arg(QString::fromStdString(session.project.model().adjustableParams[sel.archive().param].name))
                           .arg(sel.archive_idx)
                           .arg(sel.size()),
                           QSize(1, i));
        ++i;
    }

    if ( savedSelection ) {
        savedSelection = false;
        ui->combo->setCurrentIndex(ui->combo->count() - 1);
        return;
    }

    if ( currentIdx < 0 )
        return;

    ui->combo->setCurrentIndex(currentData.width() * session.wavegen().archives().size() + currentData.height());
}

void WavegenFitnessMapper::updateDimensions()
{
    if ( savedSelection || ui->combo->count() == 0 || ui->combo->currentIndex() < 0 )
        return;
    const WavegenSelection *sel = nullptr;
    const Wavegen::Archive *archive;
    QSize collection = ui->combo->currentData().toSize();
    if ( collection.width() ) {
        sel =& session.wavegenselector().selections().at(collection.height());
        archive =& sel->archive();
    } else {
        archive =& session.wavegen().archives().at(collection.height());
    }

    int i = 0, n = archive->searchd.mapeDimensions.size();
    size_t multiplier = Wavegen::mape_multiplier(archive->precision);
    groupx = new QButtonGroup(this);
    groupy = new QButtonGroup(this);
    mins.resize(n);
    maxes.resize(n);
    collapse.resize(n);
    ui->dimensions->setRowCount(n);
    QStringList labels;
    for ( MAPEDimension const& d : archive->searchd.mapeDimensions ) {
        labels << QString::fromStdString(toString(d.func));

        QRadioButton *x = new QRadioButton();
        groupx->addButton(x, i);
        ui->dimensions->setCellWidget(i, 0, x);

        QRadioButton *y = new QRadioButton();
        groupy->addButton(y, i);
        ui->dimensions->setCellWidget(i, 1, y);

        QCheckBox *c = new QCheckBox();
        ui->dimensions->setCellWidget(i, 4, c);
        collapse[i] = c;

        double step = (d.max - d.min) / (multiplier * d.resolution);
        int decimals = 1 - log10(step);

        QDoubleSpinBox *min = new QDoubleSpinBox();
        min->setDecimals(decimals);
        min->setSingleStep(step);
        min->setRange(d.min, d.max);
        min->setValue(d.min);
        ui->dimensions->setCellWidget(i, 2, min);
        mins[i] = min;

        QDoubleSpinBox *max = new QDoubleSpinBox();
        max->setDecimals(decimals);
        max->setSingleStep(step);
        max->setRange(d.min, d.max);
        max->setValue(d.max);
        ui->dimensions->setCellWidget(i, 3, max);
        maxes[i] = max;

        if ( sel ) {
            // Set selection parameters; ensure that min/max aren't rounded the wrong way by placing the values in the center of the bin
            min->setValue(sel->rmin(i) + step/2);
            max->setValue(sel->rmax(i) + step/2);
            c->setChecked(sel->ranges.at(i).collapse);
        }
        if ( !sel || sel->width(i) > 1 ) {
            if ( groupx->checkedId() < 0 )
                x->setChecked(true);
            else if ( groupy->checkedId() < 0 )
                y->setChecked(true);
        }

        ++i;
    }
    ui->dimensions->setVerticalHeaderLabels(labels);

    replot();
}

bool WavegenFitnessMapper::select(bool flattenToPlot)
{
    QSize collection = ui->combo->currentData().toSize();
    if ( collection.width() ) {
        selection.reset(new WavegenSelection(session.wavegenselector().selections().at(collection.height())));
    } else {
        selection.reset(new WavegenSelection(session, collection.height()));
    }
    WavegenSelection &sel = *selection; // For autocompletion in QtCreator

    for ( int i = 0; i < ui->dimensions->rowCount(); i++ ) {
        bool flatten = collapse[i]->isChecked();
        if ( flattenToPlot )
            flatten |= !(groupx->checkedId() == i || groupy->checkedId() == i);
        sel.limit(i, mins[i]->value(), maxes[i]->value(), flatten);
    }
    sel.finalise();
    return true;
}

void WavegenFitnessMapper::replot()
{
    int x = groupx->checkedId(), y = groupy->checkedId();
    if ( x == y || x < 0 || y < 0 )
        return;

    if ( !select(true) )
        return;

    MAPEDimension const& dimx = selection->archive().searchd.mapeDimensions[x];
    MAPEDimension const& dimy = selection->archive().searchd.mapeDimensions[y];
    size_t multiplier = Wavegen::mape_multiplier(selection->archive().precision);

    // Set up axes
    ui->plot->xAxis->setLabel(QString::fromStdString(toString(dimx.func)));
    ui->plot->yAxis->setLabel(QString::fromStdString(toString(dimy.func)));

    // set up the QCPColorMap:
    colorMap->data()->clear();
    colorMap->data()->setSize(dimx.resolution * multiplier, dimy.resolution * multiplier);
    if ( selection->ranges.at(x).collapse )
        colorMap->data()->setKeySize(1);
    if ( selection->ranges.at(y).collapse )
        colorMap->data()->setValueSize(1);
    colorMap->data()->setRange(QCPRange(dimx.min, dimx.max), QCPRange(dimy.min, dimy.max));
    // now we assign some data, by accessing the QCPColorMapData instance of the color map:
    // Note, plot area spans the full dimensional range, but only selection is assigned
    for ( std::vector<size_t> idx(selection->archive().searchd.mapeDimensions.size()); idx[x] < selection->width(x); ++idx[x] ) {
        for ( idx[y] = 0; idx[y] < selection->width(y); ++idx[y] ) {
            bool ok;
            auto it = selection->data_relative(idx, &ok);
            colorMap->data()->setCell(
                idx[x] + selection->ranges.at(x).min,
                idx[y] + selection->ranges.at(y).min,
                ok ? it->stats.fitness : 0);
        }
    }

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    ui->plot->rescaleAxes();
    ui->plot->replot();
}

void WavegenFitnessMapper::on_btnAdd_clicked()
{
    if ( !select(false) )
        return;
    savedSelection = true;
    session.wavegenselector().save(*selection);
}
