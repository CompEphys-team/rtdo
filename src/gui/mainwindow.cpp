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
                case 250:
                case 500:
                    std::cout << "Increasing resolution" << std::endl;
                    return true;
                default:
                    return false;
                }
            };

            wd.historySize = 20;

            Wavegen wg(mt, sd, wd);

            wg.r.stopFunc = [&wg](const MAPEStats &S){
                std::cout << "Search, iteration " << S.iterations << ": " << S.histIter->insertions << " insertions, population "
                          << S.population << ", best fitness: " << S.bestWave->stats.fitness << endl;
                return !S.historicInsertions || S.iterations == 1000;
            };

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
        }
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
