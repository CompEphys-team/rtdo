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

    ModelData md;
    md.dt = 0.25;
    md.method = IntegrationMethod::RungeKutta4;
    MetaModel mt(fname, md);

    StimulationData sd;
    WavegenData wd;
    wd.permute = false;
    wd.numWavesPerEpoch = 10000;
    wd.numSigmaAdjustWaveforms = 1e5;
    wd.nInitialWaves = 1e5;
    RunData rund;
    rund.accessResistance = 15;
    rund.clampGain = 1000;
    rund.simCycles = 20;
    scalar maxCycles = 100.0 / mt.cfg.dt * rund.simCycles;
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

//    Wavegen wg(mt, dir, sd, wd, rund);

//    //wg.permute();
//    wg.adjustSigmas();
/*    std::vector<MAPElite> winners;
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
    }*/

    ExperimentData expd;
    expd.numCandidates = 10000;
    expd.settleDuration = 100;

    ExperimentLibrary explib(mt, dir, expd, rund);

    ErrorProfiler errp(explib);

    Stimulation foo {};
    foo.duration = 50;
    foo.baseV = -60;
    foo.insert(foo.end(), Stimulation::Step{30, -20, false});
    foo.tObsBegin = 20;
    foo.tObsEnd = 50;

    std::cout << std::endl;
    std::cout << "Error profiles - comparing CPU (base model) and GPU (perturbated) simulations" << std::endl;
    std::cout << "Stimulating with:" << std::endl;
    std::cout << foo << std::endl;
    std::cout << "Pre-stimulation settling: " << expd.settleDuration << " ms." << std::endl;
    std::cout << "Base model (B1_basic): (base, min, max)" << std::endl;
    for ( const AdjustableParam &p : mt.adjustableParams )
        std::cout << p.name << '\t' << p.initial << '\t' << p.min << '\t' << p.max << std::endl;
    std::cout << "Each parameter perturbed in turn, distributed uniformly in [min,max] with " << expd.numCandidates << " distinct values." << std::endl;
    std::cout << "Error denotes deviation from CPU-computed base model." << std::endl << std::endl;

    std::vector<ErrorProfiler::Permutation> perms;
    for ( size_t par = 0, other; par < mt.adjustableParams.size(); par++ ) {
        perms.clear();
        perms.resize(mt.adjustableParams.size());
        perms[par].n = 100;
        perms[other = (par ? par-1 : mt.adjustableParams.size()-1)].n = 10;
        errp.setPermutations(perms);
        errp.profile(foo);
        auto profs = errp.getProfiles(par);

        std::cout << "# Total error for perturbed parameter:" << std::endl;
        std::cout << mt.adjustableParams[other].name << " -->\t";
        for ( size_t i = 0; i < profs.size(); i++ )
            std::cout << errp.getParameterValue(other, profs[i].paramIndex(other)) << " (" << profs[i].paramIndex(other) << ")" << '\t';
        std::cout << std::endl;
        std::cout << mt.adjustableParams[par].name << "\tError" << std::endl;
        for ( size_t j = 0; j < profs[0].size(); j++ ) {
            std::cout << errp.getParameterValue(par, j);
            for ( size_t i = 0; i < profs.size(); i++ )
                std::cout << '\t' << profs[i][j];
            std::cout << std::endl;
        }
        std::cout << std::endl;

        size_t parIdx = errp.getParameterIndex(par, mt.adjustableParams[par].initial);
        size_t otherIdx = errp.getParameterIndex(other, mt.adjustableParams[other].initial);
        std::cout << "# Error for initial " << mt.adjustableParams[par].name << "=" << errp.getParameterValue(par, parIdx)
                  << " with " << mt.adjustableParams[other].name << "=" << errp.getParameterValue(other, otherIdx)
                  << " is " << profs[otherIdx][parIdx] << std::endl;
        std::cout << std::endl;
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}
