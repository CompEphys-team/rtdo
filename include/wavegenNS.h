/*--------------------------------------------------------------------------
Author: Felix Kern

Institute: School of Life Sciences
University of Sussex
Falmer, Brighton BN1 9QG, UK

email to:  fbk21@sussex.ac.uk

initial version: 2016-02-16

--------------------------------------------------------------------------*/
#ifndef WAVEGENNS_H
#define WAVEGENNS_H
#include <iostream>
#include "shared.h"
#include "config.h"

class WavegenNSVirtual
{
public:
    WavegenNSVirtual(conf::Config *cfg) : cfg(cfg) {}
    virtual ~WavegenNSVirtual() {}

    virtual void runAll(std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag) = 0;
    virtual void adjustSigmas() = 0;
    virtual void noveltySearch(bool *stopFlag) = 0;
    virtual void optimiseAll(std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag) = 0;
    virtual void optimise(int param, bool *stopFlag) = 0;
    virtual void validate(inputSpec &stim, int param, std::ostream &currentfile) = 0;

    //!< Validate @arg stim, returning currents of the original and detuned models in @arg Isyns, and model-defined currents in @arg modelCurrents.
    //! @arg stim is populated with fitness for the proposed (original) observation window.
    //! @return A copy of @arg stim with observation window and fitness set by algorithmic choice for the parameter indicated in @arg param.
    //! Does not affect the database of evolved models.
    virtual inputSpec validate(inputSpec &stim, vector<vector<double>> &Isyns, vector<vector<double>> &modelCurrents, int param = 0) = 0;

protected:
    conf::Config *cfg;
};

#endif // WAVEGENNS_H
