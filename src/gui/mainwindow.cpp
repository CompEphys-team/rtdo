#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "metamodel.h"
#include <QFileDialog>
#include <sstream>
#include <fstream>
#include <iostream>
#include <dlfcn.h>
#include "kernelhelper.h"
#include "wavegen.h"

using std::endl;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    QString fname = QFileDialog::getOpenFileName();
    MetaModel mt(fname.toStdString());
    mt.cfg.dt = 0.25;
    mt.cfg.npop = 10000;
    mt.cfg.method = IntegrationMethod::RungeKutta4;
    mt.cfg.type = ModuleType::Wavegen;
    mt.cfg.permute = false;
    genn_target_generator =& mt;
    std::string dir = QFileDialog::getExistingDirectory().toStdString();
    char* argv[2];
    argv[0] = "";
    argv[1] = const_cast<char*>(dir.c_str());
    if ( !generateAll(2, argv) ) {
        dir += std::string("/") + genn_target_generator->name() + "_CODE";
        std::ofstream makefile(dir + "/Makefile", std::ios_base::app);
        makefile << endl;
        makefile << "runner.so: runner.o" << endl;
        makefile << "\t$(CXX) -o $@ $< -shared" << endl;

        std::stringstream cmd;
        cmd << "cd " << dir << " && GENN_PATH=" << LOCAL_GENN_PATH << " make runner.so";
        if ( !system(cmd.str().c_str()) ) {
            dlerror();
            void *lib;
            if ( ! (lib = dlopen((dir + "/runner.so").c_str(), RTLD_NOW)) ) {
                std::cerr << dlerror() << endl;
            }
            GeNN_Bridge::init(mt);
            StimulationData sd;
            WavegenData wd;
            wd.accessResistance = 15;
            wd.clampGain = 1000;
            wd.simCycles = 20;
            wd.numSigmaAdjustWaveforms = 20;

            Wavegen wg(mt, sd, wd);
            wg.search();
            //wg.permute();
            wg.adjustSigmas();
//            wg.adjustSigmas();
//            wg.adjustSigmas();
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
