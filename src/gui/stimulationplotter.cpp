#include "stimulationplotter.h"
#include "ui_stimulationplotter.h"
#include "colorbutton.h"
#include "stimulationgraph.h"

StimulationPlotter::StimulationPlotter(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::StimulationPlotter),
    resizing(false),
    rebuilding(false),
    enslaved(false)
{
    ui->setupUi(this);
    ui->table->horizontalHeader()->setSectionResizeMode(QHeaderView::Stretch);
    connect(ui->table->verticalHeader(), SIGNAL(sectionResized(int,int,int)), this, SLOT(resizeTableRows(int,int,int)));
    connect(ui->columns, SIGNAL(valueChanged(int)), this, SLOT(setColumnCount(int)));
    connect(ui->sources, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->nStims, SIGNAL(valueChanged(int)), this, SLOT(replot()));
    connect(ui->offset, SIGNAL(valueChanged(int)), this, SLOT(replot()));

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
        for ( QCustomPlot *p : plots ) {
            p->xAxis->setTicks(on);
            p->yAxis->setTicks(on);
            p->xAxis->setTickLabels(on);
            p->yAxis->setTickLabels(on);
            p->replot();
        }
    });
    connect(ui->titles, &QCheckBox::toggled, [=](bool on) {
        int i = 0;
        for ( QCustomPlot *p : plots ) {
            if ( on ) {
                QCPTextElement *title = new QCPTextElement(p, ui->legend->verticalHeaderItem(i++)->text());
                p->plotLayout()->insertRow(0);
                p->plotLayout()->addElement(0, 0, title);
            } else {
                p->plotLayout()->removeAt(p->plotLayout()->rowColToIndex(0,0));
                p->plotLayout()->simplify();
            }
            p->replot();
        }
    });

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
    for ( QCustomPlot *p : plots )
        delete p;
}

void StimulationPlotter::resizeTableRows(int, int, int size)
{
    if ( resizing )
        return;
    resizing = true;
    for ( int i = 0; i < ui->table->rowCount(); i++ )
        ui->table->verticalHeader()->resizeSection(i, size);
    resizing = false;
}

void StimulationPlotter::setColumnCount(int n)
{
    ui->table->clear();
    ui->table->setRowCount(std::ceil(double(plots.size())/n));
    ui->table->setColumnCount(n);
    int h = ui->table->rowHeight(0);
    size_t i = 0;
    for ( int row = 0; row < ui->table->rowCount(); row++ ) {
        ui->table->setRowHeight(row, h);
        for ( int col = 0; col < ui->table->columnCount(); col++ ) {
            QWidget *widget = new QWidget();
            QGridLayout *layout = new QGridLayout(widget);
            layout->addWidget(plots[i]);
            layout->setMargin(0);
            widget->setLayout(layout);
            ui->table->setCellWidget(row, col, widget);
            if ( ++i >= plots.size() )
                row = col = plots.size();
        }
    }
}

void StimulationPlotter::init(Session *session)
{
    this->session = session;
    connect(&session->wavesets(), SIGNAL(addedSet()), this, SLOT(updateSources()));
    updateSources();
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
    for ( QCustomPlot *p : plots )
        delete p;
    plots.resize(upper-lower);
    ui->overlay->clearGraphs();
    ui->overlay->clearItems();

    for ( size_t i = lower, row = 0; i < upper; i++, row++ ) {
        new StimulationGraph(ui->overlay->xAxis, ui->overlay->yAxis, stims[i]); // ui->overlay takes ownership
        if ( !single ) {
            plots[row] = new QCustomPlot();
            plots[row]->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
            if ( ui->titles->isChecked() ) {
                QCPTextElement *title = new QCPTextElement(plots[row], labels.at(row));
                plots[row]->plotLayout()->insertRow(0);
                plots[row]->plotLayout()->addElement(0, 0, title);
            }
            bool scale = ui->scale->isChecked();
            plots[row]->xAxis->setTicks(scale);
            plots[row]->yAxis->setTicks(scale);
            plots[row]->xAxis->setTickLabels(scale);
            plots[row]->yAxis->setTickLabels(scale);
            new StimulationGraph(plots[row]->xAxis, plots[row]->yAxis, stims[i]);
            duration = std::max(duration, double(stims[i].duration));
        }
        updateColor(row, false);
    }
    ui->overlay->xAxis->setRange(0, duration);
    ui->overlay->yAxis->setRange(minV, maxV);
    ui->overlay->replot();

    for ( QCustomPlot *p : plots ) {
        p->xAxis->setRange(0, duration);
        p->yAxis->setRange(minV, maxV);
        p->replot();
    }

    ui->columns->setMaximum(stims.size());
    setColumnCount(ui->columns->value());

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
        plots[idx]->graph()->setPen(pen);
        plots[idx]->graph()->setBrush(brush);
        if ( replot )
            plots[idx]->replot();
    }
}
