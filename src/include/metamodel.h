#ifndef METAMODEL_H
#define METAMODEL_H

#include <string>
#include <vector>
#include "types.h"
#include "../tinyxml2/tinyxml2.h"

class neuronModel;
class NNmodel;

class MetaModel
{
public:
    MetaModel(Project const& p, std::string file = "");

    neuronModel generate(NNmodel &m, std::vector<double> &fixedParamIni, std::vector<double> &variableIni) const;

    std::string supportCode() const;
    std::string populateStructs(std::string paramPre = "$(", std::string paramPost = ")",
                                std::string rundPre = "$(", std::string rundPost = ")") const;
    std::string extractState(std::string pre = "$(", std::string post = ")") const;
    std::string kernel(const std::string &tab, bool wrapVariables, bool defineCurrents) const;
    std::string daqCode(int ordinal) const;

    std::string name() const; //!< Model name, no suffix
    std::string name(ModuleType) const; //!< Model name with type suffix

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    struct Current {
        std::string name;
        std::string popen;
        scalar gUnit = 0.02;
        Variable *gbar;
        Variable *E;
    };
    std::vector<Current> currentDefs;
    StateVariable *V = nullptr;
    Variable *C = nullptr;

    static void (*modelDef)(NNmodel&);

    static size_t numLibs; //!< Count the number of open Wavegen/Experiment libraries

protected:
    const Project &project;
    std::string _name;
    std::vector<Variable> _params;

    bool isCurrent(const Variable &tmp) const;
    std::string resolveCode(const std::string &code) const;
    std::string structDeclarations() const;

    void readVariables(const tinyxml2::XMLElement *);
    void readParams(const tinyxml2::XMLElement *);
    void readAdjustableParams(const tinyxml2::XMLElement *);
    void readCurrents(const tinyxml2::XMLElement *);
    void readVoltage(const tinyxml2::XMLElement *);
    void readCapacitance(const tinyxml2::XMLElement *);
};

// generateALL.cc's renamed main - see core/generateAllNoMain.cpp:
int generateAll(int argc, char *argv[]);

// The model definition that GeNN expects to be dropped into genn-buildmodel.sh:
inline void modelDefinition(NNmodel &n) { MetaModel::modelDef(n); }

// A copy of generateAll that correctly invokes the modified genNeuronKernel
// See core/generateAllNoMain.cpp
void generateCode(std::string path, const MetaModel &metamodel);

#endif // METAMODEL_H
