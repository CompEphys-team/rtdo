#ifndef CONFIG_H
#define CONFIG_H

#include "types.h"
#include <functional>

namespace Config
{
void init();

extern ModelData Model;
extern StimulationData Stimulation;
extern WavegenLibraryData WavegenLibrary;
extern WavegenData Wavegen;
extern RunData Run;
extern ExperimentData Experiment;

extern int LOADED_PROTOCOL_VERSION;

/**
 * @brief readProtocol reads a protocol file, taking into account config version
 * @param is - the protocol file
 * @param callback - pointer to a function that is called when a parameter can't be found. Receives the
 * raw parameter name as a QString argument. May return true to cancel reading.
 * @return true if everything went to plan; false when cancelled through callback or if protocol is malformed.
 */
bool readProtocol(std::istream &is, std::function<bool(std::string)> *callback = nullptr);

/**
 * @brief writeProtocol writes the present configuration to a file.
 * @param os - An open file or other output stream.
 */
void writeProtocol(std::ostream &os);

}

#endif // CONFIG_H
