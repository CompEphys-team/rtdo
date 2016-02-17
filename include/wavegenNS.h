#ifndef WAVEGENNS_H
#define WAVEGENNS_H
#include <iostream>
#include "shared.h"
#include "config.h"

class WavegenNSVirtual
{
public:
    WavegenNSVirtual(conf::WaveGenConfig *cfg) : cfg(cfg) {}
    virtual ~WavegenNSVirtual() {}

    virtual void runAll(std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag) = 0;
    virtual void adjustSigmas() = 0;
    virtual void noveltySearch(bool *stopFlag) = 0;
    virtual void optimiseAll(std::ostream &wavefile, std::ostream &currentfile, bool *stopFlag) = 0;
    virtual void optimise(int param, bool *stopFlag) = 0;
    virtual void validate(inputSpec &stim, int param, std::ostream &currentfile) = 0;

protected:
    conf::WaveGenConfig *cfg;
};

#endif // WAVEGENNS_H
