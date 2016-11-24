#include "types.h"


bool Stimulation::operator==(const Stimulation &other) const
{
    return duration  == other.duration
        && tObsBegin == other.tObsBegin
        && tObsEnd   == other.tObsEnd
        && baseV     == other.baseV
        && steps     == other.steps;
}

bool Stimulation::Step::operator==(const Stimulation::Step &other) const
{
    return t    == other.t
        && V    == other.V
        && ramp == other.ramp;
}
