#ifndef METAMODEL_H
#define METAMODEL_H

#include <string>
#include <vector>
#include "types.h"

class neuronModel;
class NNmodel;

class MetaModel
{
public:
    MetaModel(std::string xmlfile);

    neuronModel generate(NNmodel &m, std::vector<double> &fixedParamIni, std::vector<double> &variableIni);
    std::string kernel(const std::string &tab, bool wrapVariables, bool defineCurrents) const;

    ModelData cfg;

    std::string name(ModuleType) const;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    static void (*modelDef)(NNmodel&);

protected:
    std::string _name;
    std::string _dir;
    std::vector<Variable> _params;
    double _baseV;

    void generateExperimentCode(NNmodel &m, neuronModel &n,
                                std::vector<double> &fixedParamIni,
                                std::vector<double> &variableIni);
    std::string generateExperimentBridge(const std::vector<Variable> &globals, const std::vector<Variable> &vars) const;

    bool isCurrent(const Variable &tmp) const;
};

// generateALL.cc's renamed main - see core/generateAllNoMain.cpp:
int generateAll(int argc, char *argv[]);

// The model definition that GeNN expects to be dropped into genn-buildmodel.sh:
inline void modelDefinition(NNmodel &n) { MetaModel::modelDef(n); }

#endif // METAMODEL_H
