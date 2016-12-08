#include "types.h"


bool Stimulation::operator==(const Stimulation &other) const
{
    return duration  == other.duration
        && tObsBegin == other.tObsBegin
        && tObsEnd   == other.tObsEnd
        && baseV     == other.baseV
            && steps     == other.steps;
}

std::ostream &operator<<(std::ostream &os, const Stimulation &I)
{
    using std::endl;
    os << "{" << endl
       << "  duration: " << I.duration << endl
       << "  observe: [" << I.tObsBegin << "," << I.tObsEnd << "]" << endl
       << "  V: " << I.baseV << endl;
    for ( const Stimulation::Step &s : I.steps )
        os << s;
    os << "}";
    return os;
}

bool Stimulation::Step::operator==(const Stimulation::Step &other) const
{
    return t    == other.t
        && V    == other.V
            && ramp == other.ramp;
}

std::ostream &operator<<(std::ostream &os, const Stimulation::Step &s)
{
    os << "  {" << (s.ramp?"ramp":"step") << " at " << s.t << " to " << s.V << "}" << std::endl;
    return os;
}

bool MAPElite::compete(MAPElite &&rhs)
{
    if ( rhs.fitness > fitness ) {
        using std::swap;
        swap(fitness, rhs.fitness);
        swap(wave, rhs.wave);
        return true;
    }
    return false;
}
