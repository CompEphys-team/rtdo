#include "profileplotter.h"
#include "ui_profileplotter.h"

ProfilePlotter::ProfilePlotter(Session &session, QWidget *parent) :
    QWidget(parent),
    ui(new Ui::ProfilePlotter),
    session(session)
{
    ui->setupUi(this);

    for ( AdjustableParam const& p : session.project.model().adjustableParams )
        ui->targetParam->addItem(QString::fromStdString(p.name));

    connect(&session, SIGNAL(actionLogged(QString,QString,QString,int)), this, SLOT(updateCombo()));
    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(updateTargets()));
    connect(ui->profile, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));
    connect(ui->targetParam, SIGNAL(currentIndexChanged(int)), this, SLOT(replot()));

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
    for ( size_t i = 0; i < session.profiler().profiles().size(); i++ )
        ui->profile->addItem(QString("Profile %1").arg(i));
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
    if ( profileNo < 0 || updatingCombo )
        return;
    ui->plot->clearGraphs();
    size_t targetParam = ui->targetParam->currentIndex();
    const ErrorProfile &profile = session.profiler().profiles().at(profileNo);
    for ( std::vector<ErrorProfile::Profile> const& wavep : profile.profiles(targetParam) ) {
        for ( const ErrorProfile::Profile &singlep : wavep ) {
            QCPGraph *graph = ui->plot->addGraph();
            for ( ErrorProfile::Iterator it = singlep.begin(); it != singlep.end(); ++it )
                graph->addData(profile.parameterValue(targetParam, it.index()), *it);
        }
    }
    ui->plot->rescaleAxes();
    ui->plot->xAxis->setRange(profile.permutations().at(targetParam).min, profile.permutations().at(targetParam).max);
    ui->plot->replot();
}
