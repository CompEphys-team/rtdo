#include "profileplotter.h"
#include "ui_profileplotter.h"
#include <QColorDialog>
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

    for ( AdjustableParam const& p : session.project.model().adjustableParams )
        ui->targetParam->addItem(QString::fromStdString(p.name));

    connect(&session.profiler(), SIGNAL(done()), this, SLOT(updateProfiles()));
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
    newPlot->legend->setVisible(true);

    auto uncheckRescale = [=](const QCPRange &) {
        ui->rescale->setChecked(false);
    };
    connect(newPlot->xAxis, static_cast<void (QCPAxis::*)(const QCPRange &)>(&QCPAxis::rangeChanged), uncheckRescale);
    connect(newPlot->yAxis, static_cast<void (QCPAxis::*)(const QCPRange &)>(&QCPAxis::rangeChanged), uncheckRescale);

    connect(newPlot, &QCustomPlot::selectionChangedByUser, [=](){
        newPlot->legend->setSelectedParts(QCPLegend::spNone);
        for ( QCPGraph *g : newPlot->selectedGraphs() ) {
            newPlot->legend->item(g->name().toInt())->setSelected(true);
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
        QPen pen(QColorDialog::standardColor(i));
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
            graph->setData(keys, values, true);
            graph->setName(QString::number(i));
            if ( j == 0 ) {
                graph->addToLegend();
            }

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

    QVector<double> keys(clusters.size());

    QSharedPointer<QCPAxisTickerText> ticker(new QCPAxisTickerText);
    for ( i = 0; i < clusters.size(); i++ ) {
        int n = clusters[i] - pstat.stats.cbegin();
        ticker->addTick(i, QString::number(n));
        keys[i] = i;
    }
    ui->stats->xAxis->setTicker(ticker);
    ui->stats->xAxis->setRange(-0.5, clusters.size()-0.5);

    ProfileStats::Statistic ProfileStats::Cluster::* stat;
    switch ( ui->statistic->currentIndex() ) {
    case 0: stat =& ProfileStats::Cluster::minValue; break;
    case 1: stat =& ProfileStats::Cluster::minError; break;
    case 2: stat =& ProfileStats::Cluster::deviation; break;
    case 3: stat =& ProfileStats::Cluster::localMinima; break;
    case 4: stat =& ProfileStats::Cluster::slope; break;
    case 5: stat =& ProfileStats::Cluster::slopeFit; break;
    case 6: // Performance index - no median or sd, special treatment
    {
        QVector<double> indices(clusters.size());
        for ( i = 0; i < clusters.size(); i++ ) {
            indices[i] = clusters[i]->index;
        }
        QCPBars *bars = new QCPBars(ui->stats->xAxis, ui->stats->yAxis);
        bars->setData(keys, indices, true);
        ui->stats->yAxis->rescale();
        ui->stats->replot();
        return;
    }
    }

    QVector<double> mean(clusters.size()), median(clusters.size()), sd(clusters.size());
    for ( i = 0; i < clusters.size(); i++ ) {
        mean[i] = (*clusters[i].*stat).mean;
        median[i] = (*clusters[i].*stat).median;
        sd[i] = (*clusters[i].*stat).sd;
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
