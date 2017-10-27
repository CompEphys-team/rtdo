#include "stimulationplotter.h"
#include "ui_stimulationplotter.h"
#include "colorbutton.h"
#include <QTimer>

StimulationPlotter::StimulationPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StimulationPlotter),
    rebuilding(false),
    enslaved(false)
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
            for ( int i = 0; i < ui->legend->rowCount(); i++ )
                ui->legend->item(i, 0)->setCheckState(on ? Qt::Checked : Qt::Unchecked);
            on = !on;
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
    updateSources();
    QTimer::singleShot(10, this, &StimulationPlotter::resizePanel);
}

void StimulationPlotter::setSource(WaveSource src)
{
    enslaved = true;
    single = false;
    source = std::move(src);
    ui->sources->setVisible(false);
    replot();
}

void StimulationPlotter::setStimulation(Stimulation src)
{
    enslaved = true;
    single = true;
    stim = std::move(src);
    ui->sources->setVisible(false);
    replot();
}

void StimulationPlotter::clear()
{
    enslaved = false;
    ui->sources->setVisible(true);
    updateSources();
}

void StimulationPlotter::updateSources()
{
    if ( enslaved )
        return;
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
    if ( rebuilding )
        return;

    std::vector<Stimulation> stims;
    double duration = 0, minV = session->qStimulationData().minVoltage, maxV = session->qStimulationData().maxVoltage;
    bool isDeck;
    if ( enslaved && single ) {
        stims.push_back(stim);
        duration = stim.duration;
        isDeck = false;
    } else if ( enslaved && !single ) {
        stims = source.stimulations();
        isDeck = source.type == WaveSource::Deck;
    } else if ( ui->sources->currentIndex() < 0 ) {
        return;
    } else {
        WaveSource src = session->wavesets().sources().at(ui->sources->currentIndex());
        stims = src.stimulations();
        isDeck = src.type == WaveSource::Deck;
    }

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
        if ( isDeck )
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

    for ( size_t i = 0; i < upper-lower; i++ ) {
        new StimulationGraph(ui->overlay->xAxis, ui->overlay->yAxis, stims[i+lower]); // ui->overlay takes ownership
        if ( !single ) {
            QCPAxisRect *axes = new QCPAxisRect(ui->panel);
            QCPAxis *xAxis = axes->axis(QCPAxis::atBottom);
            QCPAxis *yAxis = axes->axis(QCPAxis::atLeft);
            xAxis->setTicks(hasScale);
            xAxis->setTickLabels(hasScale);
            xAxis->setLabel(hasScale ? "Time (ms)" : "");
            xAxis->setLayer("axes");
            xAxis->grid()->setLayer("grid");
            yAxis->setTicks(hasScale);
            yAxis->setTickLabels(hasScale);
            yAxis->setLabel(hasScale ? "Voltage (mV)" : "");
            yAxis->setLayer("axes");
            yAxis->grid()->setLayer("grid");

            int row = 2 * int(i / ui->columns->value());
            int col = i % ui->columns->value();
            QCPTextElement *title = new QCPTextElement(ui->panel, labels.at(i));
            title->setVisible(hasTitle);
            ui->panel->plotLayout()->addElement(row, col, title);
            ui->panel->plotLayout()->addElement(row+1, col, axes); // add axes to panel

            graphs[i] = new StimulationGraph(xAxis, yAxis, stims[i+lower], !ui->tails->isChecked()); // add new stimGraph to axes

            duration = std::max(duration, double(stims[i+lower].duration));
        }
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
    c.setAlpha(c.alpha()/5.0);
    QBrush brush(c);
    ui->overlay->graph(idx)->setPen(pen);
    ui->overlay->graph(idx)->setBrush(brush);
    if ( replot )
        ui->overlay->replot();
    if ( !single ) {
        graphs[idx]->setPen(pen);
        graphs[idx]->setBrush(brush);
        if ( replot )
            ui->panel->replot();
    }
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
