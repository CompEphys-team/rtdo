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
#include "mapedimension.h"

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
            wd.nInitialWaves = 1000;
            wd.fitnessFunc = [](const WaveStats &S){
                double longest, abs, rel;
                if ( S.bubbles ) {
                    longest = S.longestBubble.abs / S.longestBubble.cycles;
                    abs = S.bestAbsBubble.abs / S.bestAbsBubble.cycles;
                    rel = S.bestRelBubble.abs / S.bestRelBubble.cycles;
                } else if ( S.buds ) {
                    longest = S.longestBud.abs / S.longestBud.cycles;
                    abs = S.bestAbsBud.abs / S.bestAbsBud.cycles;
                    rel = S.bestRelBud.abs / S.bestRelBud.cycles;
                } else {
                    return -__DBL_MAX__;
                }
                if ( longest > abs )
                    return longest > rel ? longest : rel;
                else
                    return abs > rel ? abs : rel;
            };
            wd.dim.push_back(std::shared_ptr<MAPEDimension>(new MAPED_numB(sd, &WaveStats::bubbles)));
            wd.dim.push_back(std::shared_ptr<MAPEDimension>(new MAPED_numB(sd, &WaveStats::buds)));
            wd.dim.push_back(std::shared_ptr<MAPEDimension>(new MAPED_BScalar(&WaveStats::longestBubble, &WaveStats::Bubble::tEnd, 0, sd.duration, 100)));
            wd.dim.push_back(std::shared_ptr<MAPEDimension>(new MAPED_BScalar(&WaveStats::longestBud, &WaveStats::Bubble::tEnd, 0, sd.duration, 100)));
            wd.historySize = 20;

            Wavegen wg(mt, sd, wd);

            wg.r.stopFunc = [&wg](const MAPEStats &S){
                std::cout << "Search, iteration " << S.iterations << ": " << S.histIter->insertions << " insertions, population "
                          << S.population << ", best fitness: " << S.bestWave->fitness << endl;
                return !S.historicInsertions;
            };

            //wg.permute();
            wg.adjustSigmas();
            wg.search(0);
            std::cout << "Search complete." << endl;
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
