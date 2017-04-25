#include "profileplotter.h"
#include "ui_profileplotter.h"
#include <QtConcurrent/QtConcurrent>

ProfilePlotter::ProfilePlotter(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ProfilePlotter),
    session(session)
{
    ui->setupUi(this);

    ui->stats->setAutoAddPlottableToLegend(false);
    ui->stats->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom);
    ui->stats->axisRect()->setRangeZoomAxes({ui->stats->yAxis});
    ui->stats->axisRect()->setRangeDragAxes({ui->stats->yAxis});
    ui->stats->xAxis->setLabel("Waveform number");

    for ( AdjustableParam const& p : session.project.model().adjustableParams )
        ui->targetParam->addItem(QString::fromStdString(p.name));

    ui->waves->setColumnWidth(0, 22);
    ui->waves->setColumnWidth(1, 25);

    ui->splitter->setStretchFactor(0, 2);
    ui->splitter->setStretchFactor(1, 1);

    connect(&session.profiler(), SIGNAL(done()), this, SLOT(updateProfiles()));
    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(updateWaves()));
    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(updateTargets()));
    connect(ui->targetParam, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->draw, SIGNAL(clicked(bool)), this, SLOT(drawProfiles()));
    connect(ui->rescale, SIGNAL(clicked(bool)), this, SLOT(rescale()));
    connect(ui->statistic, SIGNAL(currentIndexChanged(int)), this, SLOT(drawStats()));
    connect(ui->sort, SIGNAL(toggled(bool)), this, SLOT(drawStats()));

    updateProfiles();
}

ProfilePlotter::~ProfilePlotter()
{
    delete ui;
}

void ProfilePlotter::updateProfiles()
{
    int currentIdx = ui->profile->currentIndex();
    ui->profile->clear();
    for ( const ErrorProfile &ep : session.profiler().profiles() )
        ui->profile->addItem(ep.prettyName());
    if ( currentIdx >= 0 )
        ui->profile->setCurrentIndex(currentIdx);
}

void ProfilePlotter::updateTargets()
{
    int currentProfile = ui->profile->currentIndex(), initialTarget, currentTarget = initialTarget = ui->targetParam->currentIndex();
    if ( currentProfile < 0 )
        return;

    // Reset parameter names
    size_t i = 0;
    for ( AdjustableParam const& p : session.project.model().adjustableParams )
        ui->targetParam->setItemText(i++, QString::fromStdString(p.name));

    const std::vector<ErrorProfile::Permutation> &perm = session.profiler().profiles().at(currentProfile).permutations();
    const QStandardItemModel* model = qobject_cast<const QStandardItemModel*>(ui->targetParam->model());
    QStandardItem* item;
    Qt::ItemFlags flags = Qt::ItemIsSelectable|Qt::ItemIsEnabled;
    for ( i = 0; i < perm.size(); i++ ) {
        item = model->item(i);
        if ( perm[i].n > 1 ) {
            item->setFlags(item->flags() | flags);
            item->setText(QString("%1 (%2)").arg(item->text()).arg(perm[i].n));
            if ( perm[currentTarget].n == 1 )
                currentTarget = i;
        } else {
            item->setFlags(item->flags() & ~flags);
        }
    }
    if ( currentTarget != initialTarget )
        ui->targetParam->setCurrentIndex(currentTarget);
    else
        replot();
}

void ProfilePlotter::updateWaves()
{
    int profileNo = ui->profile->currentIndex();
    if ( profileNo < 0 )
        return;

    const std::vector<Stimulation> &stims = session.profiler().profiles().at(profileNo).stimulations();
    ui->waves->clearContents();
    includes.resize(stims.size());
    colors.resize(stims.size());
    ui->waves->setRowCount(stims.size());
    QStringList labels;
    QTableWidgetItem blank;
    for ( size_t i = 0; i < stims.size(); i++ ) {
        labels << QString::number(i);

        QCheckBox *box = new QCheckBox();
        QWidget *widget = new QWidget();
        QHBoxLayout *layout = new QHBoxLayout(widget);
        layout->addWidget(box);
        layout->setAlignment(Qt::AlignCenter);
        layout->setMargin(0);
        widget->setLayout(layout);
        ui->waves->setCellWidget(i, 0, widget);
        includes[i] = box;
        box->setChecked(true);
        connect(box, &QCheckBox::toggled, [=](bool on){
            includeWave(i, on);
        });

        ColorButton *btn = new ColorButton();
        ui->waves->setCellWidget(i, 1, btn);
        btn->setColor(QColorDialog::standardColor(i));
        colors[i] = btn;
        connect(btn, &ColorButton::colorChanged, [=](QColor color){
            paintWave(i, color);
        });

        // Stats are populated in drawStats()
        ui->waves->setItem(i, ValueColumn, new QTableWidgetItem(blank));
        ui->waves->setItem(i, ValueColumn+1, new QTableWidgetItem(blank));
        ui->waves->setItem(i, ValueColumn+2, new QTableWidgetItem(blank));
    }
    ui->waves->setVerticalHeaderLabels(labels);
}

void ProfilePlotter::replot()
{
    if ( ui->profile->currentIndex() < 0 || ui->targetParam->currentIndex() < 0 )
        return;
    drawStats();
    drawProfiles();
}

void ProfilePlotter::clearProfiles()
{
    QCustomPlot *newPlot = new QCustomPlot();
    newPlot->setAutoAddPlottableToLegend(false);
    newPlot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes | QCP::iSelectPlottables );

    auto uncheckRescale = [=](const QCPRange &) {
        ui->rescale->setChecked(false);
    };
    connect(newPlot->xAxis, static_cast<void (QCPAxis::*)(const QCPRange &)>(&QCPAxis::rangeChanged), uncheckRescale);
    connect(newPlot->yAxis, static_cast<void (QCPAxis::*)(const QCPRange &)>(&QCPAxis::rangeChanged), uncheckRescale);

    connect(newPlot, &QCustomPlot::selectionChangedByUser, [=](){
        for ( QCPGraph *g : newPlot->selectedGraphs() ) {
            ui->waves->selectRow(g->name().toInt());
        }
    });

    connect(newPlot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = newPlot->selectedAxes();
        if ( axes.isEmpty() )
           axes = newPlot->axisRect()->axes();
        newPlot->axisRect()->setRangeZoomAxes(axes);
        newPlot->axisRect()->setRangeDragAxes(axes);
    });
    newPlot->axisRect()->setRangeZoomAxes(newPlot->axisRect()->axes());
    newPlot->axisRect()->setRangeDragAxes(newPlot->axisRect()->axes());
    newPlot->xAxis->setLabel("Candidate model's target parameter value");
    newPlot->yAxis->setLabel("Error");

    delete ui->tab_profile->layout()->replaceWidget(ui->plot, newPlot);
    newPlot->xAxis->setRange(ui->plot->xAxis->range());
    newPlot->yAxis->setRange(ui->plot->yAxis->range());
    std::swap(ui->plot, newPlot);
    QtConcurrent::run([=](){ delete newPlot; }); // Delete asynchronously - many-graph deletion is stupendously slow!
}

void ProfilePlotter::drawProfiles()
{
    int profileNo = ui->profile->currentIndex();
    int targetParam = ui->targetParam->currentIndex();
    if ( profileNo < 0 || targetParam < 0 || !ui->draw->isChecked() )
        return;
    clearProfiles();
    ui->draw->setText("Drawing...");
    const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
    int i = 0, j, k;
    auto allProfiles = profile.profiles(targetParam);
    double ifac = 100.0/allProfiles.size(), jfac;
    for ( std::vector<ErrorProfile::Profile> const& wavep : allProfiles ) {
        j = 0;
        jfac = ifac/wavep.size();
        QPen pen(colors[i]->color);
        bool visible = includes[i]->isChecked();
        for ( const ErrorProfile::Profile &singlep : wavep ) {
            QVector<double> keys(singlep.size()), values(singlep.size());
            k = 0;
            for ( ErrorProfile::Iterator it = singlep.begin(); it != singlep.end(); ++it ) {
                keys[k] = profile.parameterValue(targetParam, it.index());
                values[k] = *it;
                k++;
            }
            QCPGraph *graph = ui->plot->addGraph();
            graph->setPen(pen);
            graph->setVisible(visible);
            graph->setData(keys, values, true);
            graph->setName(QString::number(i));

            ui->draw->setText(QString("Drawing... %1%").arg(int(i*ifac + ++j*jfac)));
            QApplication::processEvents();
            if ( profileNo != ui->profile->currentIndex() || targetParam != ui->targetParam->currentIndex() )
                return; // selection changed = new replot ??queued/under way = drop this replot
            if ( !ui->draw->isChecked() ) {
                rescale();
                ui->draw->setText("Draw");
                return;
            }
        }
        ++i;
        ui->plot->replot();
    }
    ui->draw->setText("Draw");
    rescale();
}

void ProfilePlotter::rescale()
{
    int profileNo = ui->profile->currentIndex();
    int targetParam = ui->targetParam->currentIndex();
    if ( profileNo >= 0 && targetParam >= 0 && ui->rescale->isChecked() ) {
        const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
        ui->plot->rescaleAxes();
        ui->plot->xAxis->setRange(profile.permutations().at(targetParam).min, profile.permutations().at(targetParam).max);
        ui->rescale->setChecked(true);
    }
    ui->plot->replot();
}

void ProfilePlotter::drawStats()
{
    int profileNo = ui->profile->currentIndex();
    int targetParam = ui->targetParam->currentIndex();
    if ( profileNo < 0 || targetParam < 0 )
        return;

    ui->stats->clearPlottables();

    const ProfileStats &pstat = session.profiler().profiles().at(profileNo).stats(targetParam);
    using Iter = decltype(pstat.stats)::const_iterator; // Oooh, I found a QtCreator syntax checking bug.
    std::vector<Iter> clusters(pstat.stats.size());
    size_t i = 0;
    for ( Iter cl = pstat.stats.cbegin(); cl != pstat.stats.cend(); ++cl, ++i )
        clusters[i] = cl;

    if ( ui->sort->isChecked() ) {
        std::sort(clusters.begin(), clusters.end(), [](Iter lhs, Iter rhs){
            return lhs->rank < rhs->rank;
        });
    }

    QVector<double> keys(clusters.size()), labels(clusters.size());

    QSharedPointer<QCPAxisTickerText> ticker(new QCPAxisTickerText);
    for ( i = 0; i < clusters.size(); i++ ) {
        int n = clusters[i] - pstat.stats.cbegin();
        ticker->addTick(i, QString::number(n));
        keys[i] = i;
        labels[i] = n;
    }
    ui->stats->xAxis->setTicker(ticker);
    ui->stats->xAxis->setRange(-0.5, clusters.size()-0.5);

    ProfileStats::Statistic ProfileStats::Cluster::* stat;
    switch ( ui->statistic->currentIndex() ) {
    case 1: stat =& ProfileStats::Cluster::minValue; break;
    case 2: stat =& ProfileStats::Cluster::minError; break;
    case 3: stat =& ProfileStats::Cluster::deviation; break;
    case 4: stat =& ProfileStats::Cluster::localMinima; break;
    case 5: stat =& ProfileStats::Cluster::slope; break;
    case 6: stat =& ProfileStats::Cluster::slopeFit; break;
    case 0: // Performance index - no median or sd, special treatment
    {
        for ( i = 0; i < clusters.size(); i++ ) {
            ui->waves->item(labels[i], ValueColumn)->setText(QString::number(clusters[i]->index));
            ui->waves->item(labels[i], ValueColumn+1)->setText(QString());
            ui->waves->item(labels[i], ValueColumn+2)->setText(QString());

            QCPBars *bar = new QCPBars(ui->stats->xAxis, ui->stats->yAxis);
            bar->setBrush(QBrush(colors[labels[i]]->color));
            bar->setPen(QPen(colors[labels[i]]->color.lighter()));
            bar->addData(i, clusters[i]->index);
        }
        ui->stats->yAxis->rescale();
        ui->stats->replot();
        ui->waves->horizontalHeaderItem(ValueColumn)->setText("Perform.");
        return;
    }
    }
    ui->waves->horizontalHeaderItem(ValueColumn)->setText("Mean");

    QVector<double> mean(clusters.size()), median(clusters.size()), sd(clusters.size());
    for ( i = 0; i < clusters.size(); i++ ) {
        mean[i] = (*clusters[i].*stat).mean;
        median[i] = (*clusters[i].*stat).median;
        sd[i] = (*clusters[i].*stat).sd;
        ui->waves->item(labels[i], ValueColumn)->setText(QString::number(mean[i]));
        ui->waves->item(labels[i], ValueColumn+1)->setText(QString::number(sd[i]));
        ui->waves->item(labels[i], ValueColumn+2)->setText(QString::number(median[i]));
    }

    QCPGraph *means = ui->stats->addGraph();
    means->setLineStyle(QCPGraph::lsNone);
    means->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssStar, Qt::blue, Qt::black, 6));
    means->setData(keys, mean, true);

    QCPErrorBars *errorBars = new QCPErrorBars(ui->stats->xAxis, ui->stats->yAxis);
    errorBars->setDataPlottable(means);
    errorBars->setData(sd);

    QCPGraph *medians = ui->stats->addGraph();
    medians->setLineStyle(QCPGraph::lsNone);
    medians->setScatterStyle(QCPScatterStyle(QCPScatterStyle::ssCircle, Qt::red, Qt::white, 5));
    medians->setData(keys, median, true);

    ui->stats->yAxis->rescale();
    ui->stats->replot();
}

void ProfilePlotter::includeWave(size_t waveNo, bool on)
{
    int profileNo = ui->profile->currentIndex(), targetParam = ui->targetParam->currentIndex();
    if ( profileNo < 0 || targetParam < 0 )
        return;
    const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
    int perWave = profile.numPermutations() / profile.permutations().at(targetParam).n;
    int end = std::min(ui->plot->graphCount(), int(perWave * (waveNo+1)));
    for ( int i = perWave * waveNo; i < end; i++ )
        ui->plot->graph(i)->setVisible(on);
    ui->plot->replot();
}

void ProfilePlotter::paintWave(size_t waveNo, QColor color)
{
    int profileNo = ui->profile->currentIndex(), targetParam = ui->targetParam->currentIndex();
    if ( profileNo < 0 || targetParam < 0 )
        return;
    const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
    int perWave = profile.numPermutations() / profile.permutations().at(targetParam).n;
    int end = std::min(ui->plot->graphCount(), int(perWave * (waveNo+1)));
    QPen pen(color);
    for ( int i = perWave * waveNo; i < end; i++ )
        ui->plot->graph(i)->setPen(pen);
    ui->plot->replot();
    drawStats();
}
