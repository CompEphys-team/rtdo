#include "config.h"
#include "AP.h"

#define RTDO_PROTOCOL_VERSION 1
#define RTDO_PROTOCOL_HEADER "#rtdo_config_version"

namespace Config
{

ModelData Model;
void addModel()
{
    addAP("Model.filepath", &Model, &ModelData::filepath);
    addAP("Model.dirpath", &Model, &ModelData::dirpath);
    addAP("Model.dt", &Model, &ModelData::dt);
    addAP("Model.method", &Model, &ModelData::method);
}

StimulationData Stimulation;
void addStim()
{
    addAP("Stimulation.baseV", &Stimulation, &StimulationData::baseV);
    addAP("Stimulation.duration", &Stimulation, &StimulationData::duration);
    addAP("Stimulation.minSteps", &Stimulation, &StimulationData::minSteps);
    addAP("Stimulation.maxSteps", &Stimulation, &StimulationData::maxSteps);
    addAP("Stimulation.minVoltage", &Stimulation, &StimulationData::minVoltage);
    addAP("Stimulation.maxVoltage", &Stimulation, &StimulationData::maxVoltage);
    addAP("Stimulation.minStepLength", &Stimulation, &StimulationData::minStepLength);
    addAP("Stimulation.muta.lCrossover", &Stimulation, &StimulationData::muta, &MutationData::lCrossover);
    addAP("Stimulation.muta.lLevel", &Stimulation, &StimulationData::muta, &MutationData::lLevel);
    addAP("Stimulation.muta.lNumber", &Stimulation, &StimulationData::muta, &MutationData::lNumber);
    addAP("Stimulation.muta.lSwap", &Stimulation, &StimulationData::muta, &MutationData::lSwap);
    addAP("Stimulation.muta.lTime", &Stimulation, &StimulationData::muta, &MutationData::lTime);
    addAP("Stimulation.muta.lType", &Stimulation, &StimulationData::muta, &MutationData::lType);
    addAP("Stimulation.muta.n", &Stimulation, &StimulationData::muta, &MutationData::n);
    addAP("Stimulation.muta.sdLevel", &Stimulation, &StimulationData::muta, &MutationData::sdLevel);
    addAP("Stimulation.muta.sdTime", &Stimulation, &StimulationData::muta, &MutationData::sdTime);
    addAP("Stimulation.muta.std", &Stimulation, &StimulationData::muta, &MutationData::std);
}

WavegenLibraryData WavegenLibrary;
void addWgLib()
{
    addAP("WavegenLibrary.numWavesPerEpoch", &WavegenLibrary, &WavegenLibraryData::numWavesPerEpoch);
    addAP("WavegenLibrary.permute", &WavegenLibrary, &WavegenLibraryData::permute);
}

WavegenData Wavegen;
void addWg()
{
    addAP("Wavegen.nInitialWaves", &Wavegen, &WavegenData::nInitialWaves);
    addAP("Wavegen.settleTime", &Wavegen, &WavegenData::settleTime);
    addAP("Wavegen.numSigmaAdjustWaveforms", &Wavegen, &WavegenData::numSigmaAdjustWaveforms);
    addAP("Wavegen.historySize", &Wavegen, &WavegenData::historySize);
    addAP("Wavegen.maxIterations", &Wavegen, &WavegenData::maxIterations);
    addAP("Wavegen.mapeDimensions[#].func", &Wavegen, &WavegenData::mapeDimensions, &MAPEDimension::func);
    addAP("Wavegen.mapeDimensions[#].min", &Wavegen, &WavegenData::mapeDimensions, &MAPEDimension::min);
    addAP("Wavegen.mapeDimensions[#].max", &Wavegen, &WavegenData::mapeDimensions, &MAPEDimension::max);
    addAP("Wavegen.mapeDimensions[#].resolution", &Wavegen, &WavegenData::mapeDimensions, &MAPEDimension::resolution);
    addAP("Wavegen.precisionIncreaseEpochs[#]", &Wavegen, &WavegenData::precisionIncreaseEpochs);

    scalar maxCycles = 100.0 / Model.dt * Run.simCycles;
    scalar maxDeviation = Stimulation.maxVoltage-Stimulation.baseV > Stimulation.baseV-Stimulation.minVoltage
            ? Stimulation.maxVoltage - Stimulation.baseV
            : Stimulation.baseV - Stimulation.minVoltage;
    Wavegen.mapeDimensions = {
        {MAPEDimension::Func::BestBubbleDuration,   0, maxCycles,               32},
        {MAPEDimension::Func::BestBubbleTime,       0, Stimulation.duration,    32},
        {MAPEDimension::Func::VoltageDeviation,     0, maxDeviation,            32}
    };
    Wavegen.precisionIncreaseEpochs = { 100, 500 };
}

RunData Run;
void addRun()
{
    addAP("Run.accessResistance", &Run, &RunData::accessResistance);
    addAP("Run.clampGain", &Run, &RunData::clampGain);
    addAP("Run.simCycles", &Run, &RunData::simCycles);
}

ExperimentData Experiment;
void addExperiment()
{
    addAP("Experiment.numCandidates", &Experiment, &ExperimentData::numCandidates);
    addAP("Experiment.settleDuration", &Experiment, &ExperimentData::settleDuration);
}



void init()
{
    addModel();
    addStim();
    addWgLib();
    addWg();
    addRun();
    addExperiment();
}



int LOADED_PROTOCOL_VERSION;

void writeProtocol(std::ostream &os)
{
    os << RTDO_PROTOCOL_HEADER << " " << RTDO_PROTOCOL_VERSION << std::endl << std::endl;

    for ( auto const& ap : AP::params() ) {
        ap->write(os);
    }
}

bool readProtocol(std::istream &is, std::function<bool(std::string)> *callback)
{
    QString name, header;
    int version = 0;
    std::vector<std::unique_ptr<AP>> deprec;

    if ( is.good() ) {
        is >> header;
        if ( is.good() && !header.isEmpty() && header == RTDO_PROTOCOL_HEADER ) {
            is >> version;
        } else {
            return false;
        }
    } else {
        return false;
    }

    LOADED_PROTOCOL_VERSION = version;

    AP *it;
    is >> name;
    while ( is.good() ) {
        bool ok = false;
        if ( (it = AP::find(name)) ) {
            it->readNow(name, is, &ok);
        } else if ( version < RTDO_PROTOCOL_VERSION ) {
            if ( (it = AP::find(name, &deprec)) )
                it->readNow(name, is, &ok);
        }
        if ( !ok && callback )
            if ( (*callback)(name.toStdString()) )
                return false;
        is >> name;
    }

    return true;
}

}
