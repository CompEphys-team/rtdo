#include "wavegenfitnessmapper.h"
#include "ui_wavegenfitnessmapper.h"
#include <QButtonGroup>
#include <QDoubleSpinBox>
#include "wavesource.h"

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

    connect(&session.wavesets(), SIGNAL(addedSet()), this, SLOT(updateCombo()));
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
    int nArchives = session.wavegen().archives().size();
    int nSelections = session.wavesets().selections().size();
    for ( int i = 0; i < nArchives + nSelections; i++ ) {
        WaveSource src(session,
                       i < nArchives ? WaveSource::Archive : WaveSource::Selection,
                       i < nArchives ? i : (i - nArchives));
        if ( i == ui->combo->count() || !(ui->combo->itemData(i).value<WaveSource>() == src) )
            ui->combo->insertItem(i, src.prettyName(), QVariant::fromValue(src));
    }

    if ( savedSelection ) {
        savedSelection = false;
        ui->combo->setCurrentIndex(ui->combo->count() - 1);
        return;
    }
}

void WavegenFitnessMapper::updateDimensions()
{
    if ( savedSelection || ui->combo->count() == 0 || ui->combo->currentIndex() < 0 )
        return;

    WaveSource src = ui->combo->currentData().value<WaveSource>();
    int i = 0, n = session.wavegenData(src.archive()->resultIndex).mapeDimensions.size();
    size_t multiplier = Wavegen::mape_multiplier(src.archive()->precision);
    groupx = new QButtonGroup(this);
    groupy = new QButtonGroup(this);
    mins.resize(n);
    maxes.resize(n);
    collapse.resize(n);
    ui->dimensions->setRowCount(n);
    QStringList labels;
    for ( MAPEDimension d : session.wavegenData(src.archive()->resultIndex).mapeDimensions ) {
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

        if ( src.selection() ) {
            // Set selection parameters; ensure that min/max aren't rounded the wrong way by placing the values in the center of the bin
            min->setValue(src.selection()->rmin(i) + step/2);
            max->setValue(src.selection()->rmax(i) + step/2);
            c->setChecked(src.selection()->ranges.at(i).collapse);
        }
        if ( !src.selection() || src.selection()->width(i) > 1 ) {
            if ( groupx->checkedId() < 0 )
                x->setChecked(true);
            else if ( groupy->checkedId() < 0 )
                y->setChecked(true);
        }

        ++i;
    }
    ui->dimensions->setVerticalHeaderLabels(labels);

    ui->minFitness->setValue(src.selection() ? src.selection()->minFitness : 0);

    replot();
}

bool WavegenFitnessMapper::select(bool flattenToPlot)
{
    WaveSource src = ui->combo->currentData().value<WaveSource>();
    if ( src.type == WaveSource::Selection ) {
        selection.reset(new WavegenSelection(*src.selection()));
    } else {
        selection.reset(new WavegenSelection(session, src.idx));
    }

    for ( int i = 0; i < ui->dimensions->rowCount(); i++ ) {
        bool flatten = collapse[i]->isChecked();
        if ( flattenToPlot )
            flatten |= !(groupx->checkedId() == i || groupy->checkedId() == i);
        selection->limit(i, mins[i]->value(), maxes[i]->value(), flatten);
    }
    selection->minFitness = ui->minFitness->value();
    selection->finalise();
    return true;
}

void WavegenFitnessMapper::replot()
{
    int x = groupx->checkedId(), y = groupy->checkedId();
    if ( x == y || x < 0 || y < 0 )
        return;

    if ( !select(true) )
        return;

    MAPEDimension dimx = session.wavegenData(selection->archive().resultIndex).mapeDimensions[x];
    MAPEDimension dimy = session.wavegenData(selection->archive().resultIndex).mapeDimensions[y];
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
    for ( std::vector<size_t> idx(session.wavegenData(selection->archive().resultIndex).mapeDimensions.size()); idx[x] < selection->width(x); ++idx[x] ) {
        for ( idx[y] = 0; idx[y] < selection->width(y); ++idx[y] ) {
            bool ok;
            auto it = selection->data_relative(idx, &ok);
            colorMap->data()->setCell(
                idx[x] + selection->ranges.at(x).min,
                idx[y] + selection->ranges.at(y).min,
                ok ? it->fitness : 0);
        }
    }

    // rescale the data dimension (color) such that all data points lie in the span visualized by the color gradient:
    colorMap->rescaleDataRange();
    if ( selection->minFitness > 0 )
        colorMap->setDataRange(QCPRange(selection->minFitness, colorMap->dataRange().upper));

    // rescale the key (x) and value (y) axes so the whole color map is visible:
    ui->plot->rescaleAxes();
    ui->plot->replot();
}

void WavegenFitnessMapper::on_btnAdd_clicked()
{
    if ( !select(false) )
        return;
    savedSelection = true;
    WavesetCreator &creator = session.wavesets();
    session.queue(creator.actorName(), creator.actionSelect, "", selection.release(), false);
}

void WavegenFitnessMapper::on_readMinFitness_clicked()
{
    ui->minFitness->setValue(colorMap->dataRange().lower);
}

void WavegenFitnessMapper::on_pdf_clicked()
{
    QString file = QFileDialog::getSaveFileName(this, "Select output file");
    if ( file.isEmpty() )
        return;
    if ( !file.endsWith(".pdf") )
        file.append(".pdf");
    ui->plot->savePdf(file, 0,0, QCP::epNoCosmetic, windowTitle(), ui->combo->currentText());
}
