/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include "wavegen.h"
#include "errorprofiler.h"
#include <QCloseEvent>
#include "wavegenfitnessmapper.h"
#include "profileplotter.h"
#include "fitinspector.h"
#include "stimulationplotter.h"
#include "stimulationcreator.h"
#include "gafittersettingsdialog.h"
#include "samplingprofileplotter.h"
#include "rundatadialog.h"
#include "wavegendatadialog.h"
#include "stimulationdatadialog.h"
#include "daqdialog.h"
#include "pcaplot.h"
#include "populationplot.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    wavegenDlg(nullptr),
    profileDlg(nullptr),
    project(nullptr),
    session(nullptr)
{
    ui->setupUi(this);
    statusBar()->addWidget(workerStatus = new QLabel());

    connect(ui->actionQuit, SIGNAL(triggered(bool)), this, SLOT(close()));

    connect(ui->menuSession, &QMenu::aboutToShow, [=](){
        static bool sessionsLoaded = false;
        if ( !sessionsLoaded && project && project->isFrozen() ) {
            sessionsLoaded = true;
            QDir sdir(project->dir() + "/sessions/");
            sdir.setFilter(QDir::Dirs | QDir::NoDotAndDotDot);
            sdir.setSorting(QDir::Name | QDir::Reversed); // Reverse to have the latest on top
            QStringList entries = sdir.entryList();
            if ( !entries.empty() ) {
                ui->menuSession->addSeparator();
                for ( QString sess :entries )
                    ui->menuSession->addAction(sess);
            }
        }
    });
    for ( QObject *item : ui->menuSession->children() )
        qobject_cast<QAction*>(item)->setData(QVariant::fromValue(1)); // For static entries set a valid data value
    connect(ui->menuSession, &QMenu::triggered, [=](QAction *action){
        if ( !action->data().isValid() && !action->isSeparator() ) { // For dynamically added entries, no data is set
            session = new Session(*project, project->dir() + "/sessions/" + action->text());
            sessionOpened();
        }
    });
}

MainWindow::~MainWindow()
{
    delete ui;
    delete wavegenDlg;
    delete profileDlg;
    delete session;
    delete project;
}

void MainWindow::on_actionWavegen_triggered()
{
    if ( !wavegenDlg ) {
        wavegenDlg = new WavegenDialog(*session);
    }
    wavegenDlg->show();
    wavegenDlg->raise();
    wavegenDlg->activateWindow();
    wavegenDlg->setWindowTitle(wavegenDlg->windowTitle() + " : " + title);
}

void MainWindow::on_actionDecks_triggered()
{
    if ( !deckWidget )
        deckWidget.reset(new DeckWidget(*session));
    deckWidget->show();
    deckWidget->raise();
    deckWidget->activateWindow();
    deckWidget->setWindowTitle(deckWidget->windowTitle() + " : " + title);
}

void MainWindow::on_actionProfiler_triggered()
{
    if ( !profileDlg ) {
        profileDlg = new ProfileDialog(*session);
    }
    profileDlg->show();
    profileDlg->raise();
    profileDlg->activateWindow();
    profileDlg->setWindowTitle(profileDlg->windowTitle() + " : " + title);
}

void MainWindow::on_actionSampling_profiler_triggered()
{
    if ( !sprofileDlg ) {
        sprofileDlg.reset(new SamplingProfileDialog(*session));
    }
    sprofileDlg->show();
    sprofileDlg->raise();
    sprofileDlg->activateWindow();
    sprofileDlg->setWindowTitle(sprofileDlg->windowTitle() + " : " + title);
}

void MainWindow::on_actionGAFitter_triggered()
{
    if ( !gaFitterWidget )
        gaFitterWidget.reset(new GAFitterWidget(*session));
    gaFitterWidget->show();
    gaFitterWidget->raise();
    gaFitterWidget->activateWindow();
    gaFitterWidget->setWindowTitle(gaFitterWidget->windowTitle() + " : " + title);
}

void MainWindow::on_actionScope_triggered()
{
    if ( !scope )
        scope.reset(new Scope(*session));
    scope->show();
    scope->raise();
    scope->activateWindow();
    scope->setWindowTitle(scope->windowTitle() + " : " + title);
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if ( session ) {
        if ( session->busy() &&
             QMessageBox::Yes != QMessageBox::question(this, "Application busy", "The application is busy. Are you sure you want to quit?")) {
            event->ignore();
            return;
        }
        session->quit();
    }
    if ( wavegenDlg )
        wavegenDlg->close();
    if ( profileDlg )
        profileDlg->close();
    if ( sprofileDlg )
        sprofileDlg->close();
    if ( deckWidget )
        deckWidget->close();
    if ( gaFitterWidget )
        gaFitterWidget->close();
    if ( scope )
        scope->close();
    event->accept();
}

void MainWindow::on_actionNew_project_triggered()
{
    if ( !project )
        project = new Project();
    if ( !projectSettingsDlg )
        projectSettingsDlg.reset(new ProjectSettingsDialog(project));
    else if ( project->isFrozen() )
        projectSettingsDlg->setProject(project);
    projectSettingsDlg->show();
    projectSettingsDlg->raise();
    projectSettingsDlg->activateWindow();
}

void MainWindow::on_actionOpen_project_triggered()
{
    if ( project && project->isFrozen() )
        return;

    QString file = QFileDialog::getOpenFileName(this, "Select project file", "", "Projects (*.dop)");
    if ( file.isEmpty() )
        return;
    project = new Project(file);
}

void MainWindow::on_actionNew_session_triggered()
{
    if ( !project || !project->isFrozen() )
        return;

    session = new Session(*project);
    sessionOpened();
}

void MainWindow::on_actionOpen_session_triggered()
{
    if ( !project || !project->isFrozen() )
        return;

    QString loc = QFileDialog::getExistingDirectory(this, "Select session directory");
    if ( loc.isEmpty() )
        return;
    session = new Session(*project, loc);
    sessionOpened();
}

void MainWindow::sessionOpened()
{
    ui->menuSession->setEnabled(false);
    ui->mainToolBar->setEnabled(true);
    ui->menuFigures->setEnabled(true);
    ui->menuSettings->setEnabled(true);
    ui->centralWidget->setEnabled(true);
    setTitle();
    ui->log->setModel(session->getLog());
    ui->log->setColumnWidth(0, 130);

    QString *paramName = new QString();
    connect(&session->wavegen(), &Wavegen::startedSearch, [=](QString action){
        *paramName = action;
        workerStatus->setText(QString("Wavegen searching (%1) ...").arg(*paramName));
    });
    connect(&session->wavegen(), &Wavegen::searchTick, [=](int e){
        workerStatus->setText(QString("Wavegen searching (%1) ... epoch %2").arg(*paramName).arg(e));
    });
    connect(&session->wavegen(), &Wavegen::done, [=]() {
        workerStatus->setText(QString("Wavegen search (%1) complete.").arg(*paramName));
    });
    connect(&session->profiler(), &ErrorProfiler::progress, [=](int nth, int total){
        workerStatus->setText(QString("ErrorProfiler %1/%2").arg(nth).arg(total));
    });
    connect(&session->profiler(), &ErrorProfiler::done, [=](){
        workerStatus->setText("ErrorProfiler complete.");
    });
    connect(&session->samplingProfiler(), &SamplingProfiler::progress, [=](int nth, int total){
        workerStatus->setText(QString("SamplingProfiler %1/%2").arg(nth).arg(total));
    });
    connect(&session->samplingProfiler(), &SamplingProfiler::done, [=](){
        workerStatus->setText("SamplingProfiler complete.");
    });
    connect(&session->gaFitter(), &GAFitter::starting, [=](){
        workerStatus->setText("GAFitter starting...");
    });
    connect(&session->gaFitter(), &GAFitter::progress, [=](quint32 e){
        workerStatus->setText(QString("GAFitter epoch %1").arg(e));
    });
    connect(&session->gaFitter(), &GAFitter::done, [=](){
        workerStatus->setText("GAFitter complete.");
    });
    connect(session, &Session::dispatchComplete, [=](){
        ui->pauseBtn->setText("[[ Pause ]]");
        ui->runBtn->setText("Run");
        ui->abort->setText("Abort");
    });
    connect(ui->log->selectionModel(), &QItemSelectionModel::selectionChanged, [=](const QItemSelection &, const QItemSelection &){
        ui->menuHistoric_settings->setEnabled(ui->log->selectionModel()->selectedRows().size() == 1);
    });
}

void MainWindow::setTitle()
{
    title = QString("%1 (%2) | %3")
            .arg(QDir(project->dir()).dirName())
            .arg(QString::fromStdString(project->model().name()))
            .arg(session->name());
    setWindowTitle(title);
}

void MainWindow::on_actionWavegen_fitness_map_triggered()
{
    WavegenFitnessMapper *figure = new WavegenFitnessMapper(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionError_profiles_triggered()
{
    ProfilePlotter *figure = new ProfilePlotter(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionFitting_Parameters_triggered()
{
    FitInspector *figure = new FitInspector(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionStimulations_triggered()
{
    StimulationPlotter *figure = new StimulationPlotter(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionStimulation_editor_triggered()
{
    StimulationCreator *sc = new StimulationCreator(*session);
    sc->setWindowTitle(sc->windowTitle() + " : " + title);
    sc->show();
}

void MainWindow::on_actionGA_Fitter_triggered()
{
    GAFitterSettingsDialog *dlg = new GAFitterSettingsDialog(*session);
    dlg->setWindowTitle(dlg->windowTitle() + " : " + title);
    dlg->show();
}

void MainWindow::on_actionSampled_profiles_triggered()
{
    SamplingProfilePlotter *figure = new SamplingProfilePlotter(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionRunData_triggered()
{
    RunDataDialog *dlg = new RunDataDialog(*session);
    dlg->setWindowTitle(dlg->windowTitle() + " : " + title);
    dlg->show();
}

void MainWindow::on_actionWavegenData_triggered()
{
    WavegenDataDialog *dlg = new WavegenDataDialog(*session);
    dlg->setWindowTitle(dlg->windowTitle() + " : " + title);
    dlg->show();
}

void MainWindow::on_actionStimulationData_triggered()
{
    StimulationDataDialog *dlg = new StimulationDataDialog(*session);
    dlg->setWindowTitle(dlg->windowTitle() + " : " + title);
    dlg->show();
}

void MainWindow::on_actionDAQData_triggered()
{
    DAQDialog *dlg = new DAQDialog(*session);
    dlg->setWindowTitle(dlg->windowTitle() + " : " + title);
    dlg->show();
}

void MainWindow::on_actionCrossload_from_other_session_triggered()
{
    QString loc = QFileDialog::getExistingDirectory(this, "Select session directory");
    if ( loc.isEmpty() )
        return;
    session->crossloadConfig(loc);
}

void MainWindow::on_abort_clicked()
{
    if ( session->busy() ) {
        ui->runBtn->setText("Run");
        ui->abort->setText("[[ Abort ]]");
        session->abort();
    }
}

void MainWindow::on_remove_clicked()
{
    QModelIndexList selectedRows = ui->log->selectionModel()->selectedRows();

    if ( gaFitterWidget ) {
        int n = 0;
        for ( int row = selectedRows.first().row(); row < selectedRows.back().row(); row++ )
            n += (session->getLog()->entry(row).actor == session->gaFitter().actorName());
        gaFitterWidget->unqueue(n);
    }

    session->getLog()->removeQueued(selectedRows.first().row(), selectedRows.back().row());
}

void MainWindow::on_runBtn_clicked()
{
    ui->runBtn->setText("[[ Run ]]");
    ui->pauseBtn->setText("Pause");
    if ( scope )
        scope->stop();
    session->resume();
}

void MainWindow::on_pauseBtn_clicked()
{
    ui->pauseBtn->setText("<< Pause >>");
    ui->runBtn->setText("[ Run ]");
    session->pause();
}

void MainWindow::on_desiccate_clicked()
{
    QString loc = QFileDialog::getExistingDirectory(this, "Select directory for task files");
    if ( loc.isEmpty() )
        return;

    QString desic = QFileDialog::getSaveFileName(this, "Select task log file");
    if ( desic.isEmpty() )
        return;

    session->desiccate(desic, loc);
    std::cout << "To run the desiccated tasks, copy the task files from " << loc << " to the session directory and run `rtdo " << desic << "`." << std::endl;
}

void MainWindow::on_load_clicked()
{
    QString desic = QFileDialog::getOpenFileName(this, "Select task log file");
    if ( desic.isEmpty() )
        return;

    session->exec_desiccated(desic, false);
}

void MainWindow::on_actionPCA_triggered()
{
    PCAPlot *figure = new PCAPlot(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionFit_as_heat_map_triggered()
{
    PopulationPlot *figure = new PopulationPlot(*session);
    figure->setWindowTitle(figure->windowTitle() + " : " + title);
    figure->show();
}

void MainWindow::on_actionWavegenData_2_triggered()
{
    int idx = ui->log->selectionModel()->currentIndex().row();
    WavegenDataDialog *dlg = new WavegenDataDialog(*session, idx);
    dlg->setWindowTitle(QString("%1 - %2 : %3").arg(idx,4,10,QChar('0')).arg(dlg->windowTitle(), title));
    dlg->show();
}

void MainWindow::on_actionStimulationData_2_triggered()
{
    int idx = ui->log->selectionModel()->currentIndex().row();
    StimulationDataDialog *dlg = new StimulationDataDialog(*session, idx);
    dlg->setWindowTitle(QString("%1 - %2 : %3").arg(idx,4,10,QChar('0')).arg(dlg->windowTitle(), title));
    dlg->show();
}

void MainWindow::on_actionRunData_2_triggered()
{
    int idx = ui->log->selectionModel()->currentIndex().row();
    RunDataDialog *dlg = new RunDataDialog(*session, idx);
    dlg->setWindowTitle(QString("%1 - %2 : %3").arg(idx,4,10,QChar('0')).arg(dlg->windowTitle(), title));
    dlg->show();
}

void MainWindow::on_actionDAQData_2_triggered()
{
    int idx = ui->log->selectionModel()->currentIndex().row();
    DAQDialog *dlg = new DAQDialog(*session, idx);
    dlg->setWindowTitle(QString("%1 - %2 : %3").arg(idx,4,10,QChar('0')).arg(dlg->windowTitle(), title));
    dlg->show();
}

void MainWindow::on_actionGA_Fitter_2_triggered()
{
    int idx = ui->log->selectionModel()->currentIndex().row();
    GAFitterSettingsDialog *dlg = new GAFitterSettingsDialog(*session, idx);
    dlg->setWindowTitle(QString("%1 - %2 : %3").arg(idx,4,10,QChar('0')).arg(dlg->windowTitle(), title));
    dlg->show();
}
