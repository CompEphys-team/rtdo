#include "profileplotter.h"
#include "ui_profileplotter.h"

ProfilePlotter::ProfilePlotter(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ProfilePlotter),
    session(session)
{
    ui->setupUi(this);
    ui->plot->setAutoAddPlottableToLegend(false);

    for ( AdjustableParam const& p : session.project.model().adjustableParams )
        ui->targetParam->addItem(QString::fromStdString(p.name));

    connect(&session, SIGNAL(actionLogged(QString,QString,QString,int)), this, SLOT(updateCombo()));
    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(updateTargets()));
    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->targetParam, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->draw, SIGNAL(clicked(bool)), this, SLOT(replot()));
    connect(ui->rescale, SIGNAL(clicked(bool)), this, SLOT(rescale()));

    auto uncheckRescale = [=](const QCPRange &) {
        ui->rescale->setChecked(false);
    };
    connect(ui->plot->xAxis, static_cast<void (QCPAxis::*)(const QCPRange &)>(&QCPAxis::rangeChanged), uncheckRescale);
    connect(ui->plot->yAxis, static_cast<void (QCPAxis::*)(const QCPRange &)>(&QCPAxis::rangeChanged), uncheckRescale);

    updateCombo();
    replot();

    ui->plot->setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectAxes);

    connect(ui->plot, &QCustomPlot::selectionChangedByUser, [=](){
        QList<QCPAxis *> axes = ui->plot->selectedAxes();
        if ( axes.isEmpty() )
           axes = ui->plot->axisRect()->axes();
        ui->plot->axisRect()->setRangeZoomAxes(axes);
        ui->plot->axisRect()->setRangeDragAxes(axes);
    });
    ui->plot->axisRect()->setRangeZoomAxes(ui->plot->axisRect()->axes());
    ui->plot->axisRect()->setRangeDragAxes(ui->plot->axisRect()->axes());
    ui->plot->xAxis->setLabel("Candidate model's target parameter value");
    ui->plot->yAxis->setLabel("Error");
}

ProfilePlotter::~ProfilePlotter()
{
    delete ui;
}

void ProfilePlotter::updateCombo()
{
    updatingCombo = true;
    int currentIdx = ui->profile->currentIndex();
    ui->profile->clear();
    for ( const ErrorProfile &ep : session.profiler().profiles() )
        ui->profile->addItem(ep.prettyName());
    if ( currentIdx >= 0 )
        ui->profile->setCurrentIndex(currentIdx);
    updatingCombo = false;
}

void ProfilePlotter::updateTargets()
{
    int currentProfile = ui->profile->currentIndex(), currentTarget = ui->targetParam->currentIndex();
    if ( currentProfile < 0 )
        return;
    updatingCombo = true;
    const std::vector<ErrorProfile::Permutation> &perm = session.profiler().profiles().at(currentProfile).permutations();
    const QStandardItemModel* model = qobject_cast<const QStandardItemModel*>(ui->targetParam->model());
    QStandardItem* item;
    Qt::ItemFlags flags = Qt::ItemIsSelectable|Qt::ItemIsEnabled;
    for ( size_t i = 0; i < perm.size(); i++ ) {
        item = model->item(i);
        item->setFlags(perm[i].n > 1 ? item->flags() | flags : item->flags() & ~flags);
        if ( perm[currentTarget].n <= 1 && perm[i].n > 1 )
            currentTarget = i;
    }
    ui->targetParam->setCurrentIndex(currentTarget);
    updatingCombo = false;
}

void ProfilePlotter::replot()
{
    int profileNo = ui->profile->currentIndex();
    if ( profileNo < 0 || updatingCombo || !ui->draw->isChecked() )
        return;
    ui->draw->setText("Clearing...");
    QApplication::processEvents();
    ui->plot->clearGraphs();
    ui->plot->replot();
    ui->draw->setText("Drawing...");
    QApplication::processEvents();
    size_t targetParam = ui->targetParam->currentIndex();
    const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
    int i = 0, j, k;
    auto allProfiles = profile.profiles(targetParam);
    double ifac = 100.0/allProfiles.size(), jfac;
    for ( std::vector<ErrorProfile::Profile> const& wavep : allProfiles ) {
        j = 0;
        jfac = ifac/wavep.size();
        for ( const ErrorProfile::Profile &singlep : wavep ) {
            QVector<double> keys(singlep.size()), values(singlep.size());
            k = 0;
            for ( ErrorProfile::Iterator it = singlep.begin(); it != singlep.end(); ++it ) {
                keys[k] = profile.parameterValue(targetParam, it.index());
                values[k] = *it;
                k++;
            }
            QCPGraph *graph = ui->plot->addGraph();
            graph->addData(keys, values, true);

            ui->draw->setText(QString("Drawing... %1%").arg(int(i*ifac + ++j*jfac)));
            QApplication::processEvents();
            if ( !ui->draw->isChecked() ) {
                rescale();
                ui->draw->setText("Draw");
                return;
            }
        }
        ++i;
    }
    ui->draw->setText("Draw");
    rescale();
}

void ProfilePlotter::rescale()
{
    int profileNo = ui->profile->currentIndex();
    int targetParam = ui->targetParam->currentIndex();
    if ( profileNo >= 0 && targetParam >= 0 && !updatingCombo && ui->rescale->isChecked() ) {
        const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
        ui->plot->rescaleAxes();
        ui->plot->xAxis->setRange(profile.permutations().at(targetParam).min, profile.permutations().at(targetParam).max);
        ui->rescale->setChecked(true);
    }
    ui->plot->replot();
}
