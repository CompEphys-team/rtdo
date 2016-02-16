#ifndef WAVEGENNS_H
#define WAVEGENNS_H
#include <iostream>
#include "shared.h"

class WavegenNSVirtual
{
public:
    WavegenNSVirtual() {}
    virtual ~WavegenNSVirtual() {}

    virtual void runAll(int nGenerationsNS, int nGenerationsOptimise, std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag) = 0;
    virtual void adjustSigmas() = 0;
    virtual void noveltySearch(int nGenerations, bool *stopFlag) = 0;
    virtual void optimiseAll(int nGenerations, std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag) = 0;
    virtual void optimise(int param, int nGenerations, bool *stopFlag) = 0;
    virtual void validate(inputSpec &stim, int param, std::ostream &currentfile) = 0;
};

#endif // WAVEGENNS_H
