#ifndef WAVEGENCONSTRUCTOR_H
#define WAVEGENCONSTRUCTOR_H

#include "metamodel.h"

class WavegenConstructor
{
public:
    WavegenConstructor(MetaModel &m, const std::string &directory);

    void GeNN_modelDefinition(NNmodel &);

protected:
    MetaModel &m;

    int numGroupsPerBlock; //!< Exposes the number of model groups interleaved in each block.
    int numGroups; //!< Exposes the total number of model groups.
    int numBlocks; //!< Exposes the total number of thread blocks, all of which are fully occupied.
    /// Note: A "group" consists of nParams+1 models: A base model, and one detuned model in each parameter.

private:
    std::string simCode();
    std::string supportCode(const std::vector<Variable> &globals, const std::vector<Variable> &vars);

    void *lib;
};

#endif // WAVEGENCONSTRUCTOR_H
