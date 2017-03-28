#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "metamodel.h"
#include <QFileDialog>
#include <sstream>
#include <fstream>
#include <iostream>
#include <dlfcn.h>
#include "wavegen.h"
#include "errorprofiler.h"
#include <QCloseEvent>
#include "wavegenfitnessmapper.h"

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




//    ExperimentLibrary explib(mt, Config::Experiment, Config::Run);

//    ErrorProfiler errp(explib);

//    Stimulation foo {};
//    foo.duration = 50;
//    foo.baseV = -60;
//    foo.insert(foo.end(), Stimulation::Step{30, -20, false});
//    foo.tObsBegin = 20;
//    foo.tObsEnd = 50;

//    std::cout << std::endl;
//    std::cout << "Error profiles - comparing CPU (base model) and GPU (perturbated) simulations" << std::endl;
//    std::cout << "Stimulating with:" << std::endl;
//    std::cout << foo << std::endl;
//    std::cout << "Pre-stimulation settling: " << Config::Experiment.settleDuration << " ms." << std::endl;
//    std::cout << "Base model (B1_basic): (base, min, max)" << std::endl;
//    for ( const AdjustableParam &p : mt.adjustableParams )
//        std::cout << p.name << '\t' << p.initial << '\t' << p.min << '\t' << p.max << std::endl;
//    std::cout << "Each parameter perturbed in turn, distributed uniformly in [min,max] with " << Config::Experiment.numCandidates << " distinct values." << std::endl;
//    std::cout << "Error denotes deviation from CPU-computed base model." << std::endl << std::endl;

//    std::vector<ErrorProfiler::Permutation> perms;
//    for ( size_t par = 0, other; par < mt.adjustableParams.size(); par++ ) {
//        perms.clear();
//        perms.resize(mt.adjustableParams.size());
//        perms[par].n = 100;
//        perms[other = (par ? par-1 : mt.adjustableParams.size()-1)].n = 10;
//        errp.setPermutations(perms);
//        errp.profile(foo);
//        auto profs = errp.getProfiles(par);

//        std::cout << "# Total error for perturbed parameter:" << std::endl;
//        std::cout << mt.adjustableParams[other].name << " -->\t";
//        for ( size_t i = 0; i < profs.size(); i++ )
//            std::cout << errp.getParameterValue(other, profs[i].paramIndex(other)) << " (" << profs[i].paramIndex(other) << ")" << '\t';
//        std::cout << std::endl;
//        std::cout << mt.adjustableParams[par].name << "\tError" << std::endl;
//        for ( size_t j = 0; j < profs[0].size(); j++ ) {
//            std::cout << errp.getParameterValue(par, j);
//            for ( size_t i = 0; i < profs.size(); i++ )
//                std::cout << '\t' << profs[i][j];
//            std::cout << std::endl;
//        }
//        std::cout << std::endl;

//        size_t parIdx = errp.getParameterIndex(par, mt.adjustableParams[par].initial);
//        size_t otherIdx = errp.getParameterIndex(other, mt.adjustableParams[other].initial);
//        std::cout << "# Error for initial " << mt.adjustableParams[par].name << "=" << errp.getParameterValue(par, parIdx)
//                  << " with " << mt.adjustableParams[other].name << "=" << errp.getParameterValue(other, otherIdx)
//                  << " is " << profs[otherIdx][parIdx] << std::endl;
//        std::cout << std::endl;
//    }
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
