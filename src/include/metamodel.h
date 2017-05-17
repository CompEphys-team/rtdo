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
    MetaModel(Project const& p);

    neuronModel generate(NNmodel &m, std::vector<double> &fixedParamIni, std::vector<double> &variableIni);

    std::string kernel(const std::string &tab, bool wrapVariables, bool defineCurrents) const;

    std::string name() const; //!< Model name, no suffix
    std::string name(ModuleType) const; //!< Model name with type suffix

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    static void (*modelDef)(NNmodel&);

    static size_t numLibs; //!< Count the number of open Wavegen/Experiment libraries

protected:
    const Project &project;
    std::string _name;
    std::vector<Variable> _params;
    double _baseV;

    bool isCurrent(const Variable &tmp) const;
};

// generateALL.cc's renamed main - see core/generateAllNoMain.cpp:
int generateAll(int argc, char *argv[]);

// The model definition that GeNN expects to be dropped into genn-buildmodel.sh:
inline void modelDefinition(NNmodel &n) { MetaModel::modelDef(n); }

#endif // METAMODEL_H
