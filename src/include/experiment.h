#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include "experimentlibrary.h"

class Experiment
{
public:
    Experiment(ExperimentLibrary &lib, DAQ *daq = nullptr);
    ~Experiment();

    /**
     * @brief errProfile generates an error profile for a given target parameter and stimulation
     * @param I is the stimulation applied to all profiled models
     * @param model is the set of parameter values (ordered like the MetaModel::adjustableParams, target parameter value is ignored)
     * @param targetParam is the index of a parameter to be varied uniformly within its range, @sa errProfile_value()
     * @return a vector of error values for the model corresponding to the target parameter distribution
     */
    std::vector<scalar> errProfile(const Stimulation &I, std::vector<scalar> model, size_t targetParam);
    /**
     * @brief errProfile_value gives a target parameter's value at a given position in its range. The values are uniformly
     * distributed across the full range of the parameter. Limits are as follows:
     * idx == 0 <==> value == parameter minimum
     * idx == expd.numCandidates-1 <==> value == parameter maximum
     * @sa errProfile()
     */
    scalar errProfile_value(size_t targetParam, size_t idx);
    /**
     * @brief errProfile_idx gives the index in a target parameter's distribution that most closely matches the given value.
     * Use this to find the profile of a specific model. Within resolution limits, x == errProfile_value(t, errProfile_idx(t, x))
     * @sa errProfile(), errProfile_value()
     */
    size_t errProfile_idx(size_t targetParam, scalar value);

    inline void errProfile_retain(std::vector<size_t> indices = std::vector<size_t>()) { errProfile_retainedIdx = indices; }
    std::vector<scalar> errProfile_getRetained(size_t idx);

    ExperimentLibrary &lib;

protected:
    DAQ *simulator;
    DAQ *daq;

    std::vector<size_t> errProfile_retainedIdx;
    std::list<std::vector<scalar>> errProfile_retained;

    void stimulate(const Stimulation &I);
};

#endif // EXPERIMENT_H
