#include "mainwindow.h"
#include "ui_mainwindow.h"
#include <QFileDialog>
#include "wavegen.h"
#include "errorprofiler.h"
#include <QCloseEvent>
#include "wavegenfitnessmapper.h"
#include "profileplotter.h"

using std::endl;

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
}

void MainWindow::on_actionProfiler_triggered()
{
    if ( !profileDlg ) {
        profileDlg = new ProfileDialog(*session);
    }
    profileDlg->show();
    profileDlg->raise();
    profileDlg->activateWindow();
}

void MainWindow::closeEvent(QCloseEvent *event)
{
    if ( session )
        session->quit();
    if ( wavegenDlg )
        wavegenDlg->close();
    if ( profileDlg )
        profileDlg->close();
    event->accept();
}

void MainWindow::on_actionNew_project_triggered()
{
    QString file = QFileDialog::getOpenFileName(this, "Select model file", "", "Model files (*.xml)");
    if ( file.isEmpty() )
        return;
    QString loc = QFileDialog::getExistingDirectory(this, "Select project location");
    if ( loc.isEmpty() )
        return;
    project = new Project();
    project->setModel(file);
    project->setLocation(loc + "/project.dop");
    project->compile();
    ui->menuProject->setEnabled(false);
    ui->menuSession->setEnabled(true);
}

void MainWindow::on_actionOpen_project_triggered()
{
    QString file = QFileDialog::getOpenFileName(this, "Select project file", "", "Projects (*.dop)");
    if ( file.isEmpty() )
        return;
    project = new Project(file);
    ui->menuProject->setEnabled(false);
    ui->menuSession->setEnabled(true);
}

void MainWindow::on_actionNew_session_triggered()
{
    session = new Session(*project);
    ui->menuSession->setEnabled(false);
    ui->mainToolBar->setEnabled(true);
    ui->menuFigures->setEnabled(true);
}

void MainWindow::on_actionOpen_session_triggered()
{
    QString loc = QFileDialog::getExistingDirectory(this, "Select session directory");
    if ( loc.isEmpty() )
        return;
    session = new Session(*project, loc);
    ui->menuSession->setEnabled(false);
    ui->mainToolBar->setEnabled(true);
    ui->menuFigures->setEnabled(true);
}

void MainWindow::on_actionWavegen_fitness_map_triggered()
{
    WavegenFitnessMapper *figure = new WavegenFitnessMapper(*session);
    figure->show();
}

void MainWindow::on_actionError_profiles_triggered()
{
    ProfilePlotter *figure = new ProfilePlotter(*session);
    figure->show();
}
