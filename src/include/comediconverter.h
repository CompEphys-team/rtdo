#ifndef COMEDICONVERTER_H
#define COMEDICONVERTER_H

#include <comedi.h>
#include "types.h"

class ComediConverter
{
public:
    ComediConverter(const ChnData &p, DAQData *c, bool isInChn);
    ComediConverter(const inChnData &p, DAQData *c);
    ComediConverter(const outChnData &p, DAQData *c);
    ~ComediConverter();

    double toPhys(lsampl_t) const;
    lsampl_t toSamp(double) const;

private:
    bool isInChn;
    bool has_cal;
    struct mycomedi_polynomial_t *polynomial;
    struct mycomedi_range *range;
    lsampl_t maxdata;
    double gain;
    double offset;
};

#endif // COMEDICONVERTER_H
