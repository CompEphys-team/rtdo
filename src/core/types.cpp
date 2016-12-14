#include "types.h"
#include <cassert>

void Stimulation::insert(Stimulation::Step *position, Stimulation::Step &&value)
{
    if ( position < steps || position > end() )
        throw std::runtime_error("Stimulation::insert: invalid position.");
    if ( numSteps == maxSteps )
        throw std::runtime_error("Stimulation::insert: Step array is full.");

    for ( Step *it = end(); it > position; it-- ) // Move everything from position to the end right by one
        *it = *(it-1);
    *position = value;
    numSteps++;
    assert(numSteps <= maxSteps);
}

void Stimulation::erase(Stimulation::Step *position)
{
    if ( position < steps || position >= end() )
        throw std::runtime_error("Stimulation::erase: invalid position.");
    for ( Step *it = position; it < end()-1; it++ ) // Move everything from position to the end left by one
        *it = *(it+1);
    numSteps--;
    assert(numSteps >= 0);
}

bool Stimulation::operator==(const Stimulation &other) const
{
    bool equal =
           duration  == other.duration
        && tObsBegin == other.tObsBegin
        && tObsEnd   == other.tObsEnd
        && baseV     == other.baseV
        && numSteps  == other.numSteps;
    for ( size_t i = 0; equal && i < size(); i++ )
        equal &= steps[i] == other.steps[i];
    return equal;
}

std::ostream &operator<<(std::ostream &os, const Stimulation &I)
{
    using std::endl;
    os << "{" << endl
       << "  duration: " << I.duration << endl
       << "  observe: [" << I.tObsBegin << "," << I.tObsEnd << "]" << endl
       << "  V: " << I.baseV << endl;
    for ( const Stimulation::Step &s : I )
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

bool MAPElite::compete(const MAPElite &rhs)
{
    if ( rhs.fitness > fitness ) {
        fitness = rhs.fitness;
        wave = rhs.wave;
        stats = rhs.stats;
        return true;
    }
    return false;
}
