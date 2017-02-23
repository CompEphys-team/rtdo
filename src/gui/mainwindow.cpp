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
#include "config.h"

using std::endl;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    wavegenDlg(nullptr),
    model(Config::Model)
{
    ui->setupUi(this);

    gthread.start();




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
}

void MainWindow::on_actionWavegen_triggered()
{
    if ( !wavegenDlg )
        wavegenDlg = new WavegenDialog(model, &gthread, this);
    wavegenDlg->show();
    wavegenDlg->raise();
}
