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

protected:
    conf::Config *cfg;
};

#endif // WAVEGENNS_H
