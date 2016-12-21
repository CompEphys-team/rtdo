#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "metamodel.h"
#include <QFileDialog>
#include <sstream>
#include <fstream>
#include <iostream>
#include <dlfcn.h>
#include "wavegen_globals.h"
#include "wavegen.h"
#include "experiment_globals.h"

using std::endl;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    std::string fname = QFileDialog::getOpenFileName().toStdString();
    std::string dir = QFileDialog::getExistingDirectory().toStdString();

    if ( fname.empty() || dir.empty() )
        return;

    MetaModel mt(fname);
    mt.cfg.dt = 0.25;
    mt.cfg.npop = 10000;
    mt.cfg.method = IntegrationMethod::RungeKutta4;
    mt.cfg.permute = false;
    StimulationData sd;
    WavegenData wd;
    wd.accessResistance = 15;
    wd.clampGain = 1000;
    wd.simCycles = 20;
    wd.numSigmaAdjustWaveforms = 1e5;
    wd.nInitialWaves = 1e5;
    scalar maxCycles = 100.0 / mt.cfg.dt * wd.simCycles;
    scalar maxDeviation = sd.maxVoltage-sd.baseV > sd.baseV-sd.minVoltage ? sd.maxVoltage-sd.baseV : sd.baseV - sd.minVoltage;
    wd.binFunc = [=](const Stimulation &I, const WaveStats &S, size_t precision){
        std::vector<size_t> ret(3);
        size_t mult = (1 << (5 + precision));
        ret[0] = mult * scalar(S.best.cycles) / maxCycles; //< Dimension 1 : Bubble duration, normalised to 100ms
        ret[1] = mult * S.best.tEnd / sd.duration; //< Dimension 2 : Time at end of bubble, normalised by total duration

        scalar prevV = sd.baseV, prevT = 0., deviation = 0.;
        for ( const Stimulation::Step &s : I ) {
            scalar sT = s.t, sV = s.V;
            if ( sT > S.best.tEnd ) { // Shorten last step to end of observed period
                sT = S.best.tEnd;
                if ( s.ramp )
                    sV = (s.V - prevV) * (sT - prevT)/(s.t - prevT);
            }
            if ( s.ramp ) {
                if ( (sV >= sd.baseV && prevV >= sd.baseV) || (sV <= sd.baseV && prevV <= sd.baseV) ) { // Ramp does not cross baseV
                    deviation += fabs((sV + prevV)/2 - sd.baseV) * (sT - prevT);
                } else { // Ramp crosses baseV, treat the two sides separately:
                    scalar r1 = fabs(prevV - sd.baseV), r2 = fabs(sV - sd.baseV);
                    scalar tCross = r1 / (r1 + r2) * (sT - prevT); //< time from beginning of ramp to baseV crossing
                    deviation += r1/2*tCross + r2/2*(sT - prevT - tCross);
                }
            } else {
                deviation += fabs(prevV - sd.baseV) * (sT - prevT); // Step only changes to sV at time sT
            }
            prevT = sT;
            prevV = sV;
            if ( s.t > S.best.tEnd )
                break;
        }
        if ( prevT < S.best.tEnd ) { // Add remainder if last step ends before the observed period
            deviation += fabs(prevV - sd.baseV) * (S.best.tEnd - prevT);
            prevT = S.best.tEnd;
        }
        ret[2] = mult * (deviation/prevT) / maxDeviation; //< Dimension 3 : Average command voltage deviation from holding potential,
                                                          // normalised by the maximum possible deviation
        return ret;
    };

    wd.increasePrecision = [](const MAPEStats &S){
        switch ( S.iterations ) {
        case 100:
        case 500:
            std::cout << "Increasing resolution" << std::endl;
            return true;
        default:
            return false;
        }
    };

    wd.stopFunc = [](const MAPEStats &S){
        std::cout << "Search, iteration " << S.iterations << ": " << S.histIter->insertions << " insertions, population "
                  << S.population << ", best fitness: " << S.bestWave->stats.fitness << endl;
        return !S.historicInsertions || S.iterations == 1000;
    };

    wd.historySize = 20;

    Wavegen wg(mt, dir, sd, wd);

    //wg.permute();
    wg.adjustSigmas();
    std::vector<MAPElite> winners;
    std::vector<MAPEStats> stats;
    for ( size_t i = 0, end = mt.adjustableParams.size(); i < end; i++ ) {
        std::cout << "Finding waveforms for param '" << mt.adjustableParams[i].name << "' (" << i << "/" << end << ")" << endl;
        wg.search(i);
        winners.push_back(*wg.mapeStats.bestWave);
        stats.push_back(wg.mapeStats);
    }
    std::cout << "Search complete." << endl;
    std::cout << "Winning waves and stats follow:" << endl;
    for ( size_t i = 0; i < mt.adjustableParams.size(); i++ ) {
        std::cout << "--- Param " << mt.adjustableParams[i].name << " ---" << endl;
        std::cout << "MAPE iterations: " << stats[i].iterations << endl;
        std::cout << "Total insertions: " << stats[i].insertions << endl;
        std::cout << "Final population: " << stats[i].population << endl;
        std::cout << "Best waveform: " << winners[i].wave << endl;
        std::cout << "Best waveform stats: " << winners[i].stats << endl;
    }

//    if ( mt.createModules(dir) ) {

//        dlerror();
//        void *explib;
//        mt.cfg.type = ModuleType::Experiment;
//        if ( !(explib = dlopen((dir + "/" + mt.name() + "_CODE/runner.so").c_str(), RTLD_NOW)) )
//            std::cerr << dlerror() << endl;
//        Experiment_Global::init(mt);
//        DAQData daqd;
//        daqd.dt = mt.cfg.dt;
//        RunData rund;
//        rund.accessResistance = 15;
//        rund.clampGain = 1e3;
//        rund.simCycles = 20;
//        DAQ *sim = Experiment_Global::createSimulator(&daqd, &rund);
//        Stimulation foo {};
//        foo.duration = 20;
//        foo.baseV = -60;
//        foo.insert(foo.end(), Stimulation::Step{5, 50, true});
//        foo.insert(foo.end(), Stimulation::Step{10, 20, true});
//        foo.insert(foo.end(), Stimulation::Step{15, 0, false});
//        foo.insert(foo.end(), Stimulation::Step{20, -60, true});
//        std::cout << foo << std::endl;
//        sim->run(foo);
//        for ( int i = 0; i < 80; i++ ) {
//            std::cout << sim->current << '\t' << sim->voltage << std::endl;
//            sim->next();
//        }
//        return;




//    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
