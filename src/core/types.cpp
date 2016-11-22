#include "types.h"


bool Stimulation::operator==(const Stimulation &other) const
{
    return t     == other.t
        && ot    == other.ot
        && dur   == other.dur
        && baseV == other.baseV
        && N     == other.N
        && st    == other.st
        && V     == other.V
        && ramp  == other.ramp;
}
