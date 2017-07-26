#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include "wavegen.h"
#include "errorprofiler.h"
#include <QCloseEvent>
#include "wavegenfitnessmapper.h"
#include "profileplotter.h"
#include "parameterfitplotter.h"
#include "stimulationplotter.h"
#include "stimulationcreator.h"
#include "gafittersettingsdialog.h"
#include "samplingprofileplotter.h"
#include "rundatadialog.h"
#include "wavegendatadialog.h"
#include "stimulationdatadialog.h"
#include "daqdialog.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    wavegenDlg(nullptr),
    profileDlg(nullptr),
    project(nullptr),
    session(nullptr)
{
    ui->setupUi(this);

    connect(ui->actionQuit, SIGNAL(triggered(bool)), this, SLOT(close()));
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

void MainWindow::closeEvent(QCloseEvent *event)
{
    if ( session )
        session->quit();
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
    ui->menuSession->setEnabled(false);
    ui->mainToolBar->setEnabled(true);
    ui->menuFigures->setEnabled(true);
    ui->menuSettings->setEnabled(true);
    setTitle();
}

void MainWindow::on_actionOpen_session_triggered()
{
    if ( !project || !project->isFrozen() )
        return;

    QString loc = QFileDialog::getExistingDirectory(this, "Select session directory");
    if ( loc.isEmpty() )
        return;
    session = new Session(*project, loc);
    ui->menuSession->setEnabled(false);
    ui->mainToolBar->setEnabled(true);
    ui->menuFigures->setEnabled(true);
    ui->menuSettings->setEnabled(true);
    setTitle();
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
    ParameterFitPlotter *figure = new ParameterFitPlotter(*session);
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
