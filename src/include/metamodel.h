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

    void generate(NNmodel &m);

    ModelData cfg;

    std::string name() const;

    std::vector<StateVariable> stateVariables;
    std::vector<AdjustableParam> adjustableParams;
    std::vector<Variable> currents;

    int numGroupsPerBlock; //!< Wavegen only: Exposes the number of model groups interleaved in each block.
    int numGroups; //!< Wavegen only: Exposes the total number of model groups.
    int numBlocks; //!< Wavegen only: Exposes the total number of thread blocks, all of which are fully occupied.
    /// Note: A "group" consists of nParams+1 models: A base model, and one detuned model in each parameter.

protected:
    std::string _name;
    std::vector<Variable> _params;
    double _baseV;

    void generateExperimentCode(NNmodel &m, neuronModel &n,
                                std::vector<double> &fixedParamIni,
                                std::vector<double> &variableIni);
    void generateWavegenCode(NNmodel &m, neuronModel &n,
                             std::vector<double> &fixedParamIni,
                             std::vector<double> &variableIni);
    std::string kernel(const std::string &tab) const;
    std::string bridge(const std::vector<Variable> &globals, const std::vector<Variable> &vars) const;

    bool isCurrent(const Variable &tmp) const;
};

extern MetaModel *genn_target_generator;

// generateALL.cc's renamed main - see core/generateAllNoMain.cpp:
int generateAll(int argc, char *argv[]);

// The model definition that GeNN expects to be dropped into genn-buildmodel.sh:
inline void modelDefinition(NNmodel &n) { genn_target_generator->generate(n); }

#endif // METAMODEL_H
