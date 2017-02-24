#include "types.h"
#include <cassert>
#include <cmath>

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
    assert(numSteps > 0);
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

std::ostream& operator<<(std::ostream& os, const IntegrationMethod &m)
{
    switch ( m ) {
    case IntegrationMethod::ForwardEuler: os << "ForwardEuler"; break;
    case IntegrationMethod::RungeKutta4: os << "RungeKutta4"; break;
    }
    return os;
}
std::istream& operator>>(std::istream& is, IntegrationMethod &m)
{
    std::string s;
    is >> s;
    if ( s == std::string("ForwardEuler") )     m = IntegrationMethod::ForwardEuler;
    else if ( s == std::string("RungeKutta4") ) m = IntegrationMethod::RungeKutta4;
    else /* Default: */ m = IntegrationMethod::RungeKutta4;
    return is;
}

bool MAPElite::compete(const MAPElite &rhs)
{
    if ( rhs.stats.fitness > stats.fitness ) {
        stats = rhs.stats;
        wave = rhs.wave;
        return true;
    }
    return false;
}

std::ostream &operator<<(std::ostream &os, const WaveStats &S)
{
    os << "{" << S.bubbles << " bubbles, best one lasting " << S.best.cycles
       << " cycles until " << S.best.tEnd << " and achieving " << S.fitness << " fitness.}" << std::endl;
     return os;
}

size_t MAPEDimension::bin(const Stimulation &I, const WaveStats &S, size_t multiplier) const
{
    scalar intermediate = 0.0;
    scalar factor = 1.0;
    switch ( func ) {
    case MAPEDimension::Func::BestBubbleDuration:
        intermediate = S.best.cycles;
        break;
    case MAPEDimension::Func::BestBubbleTime:
        intermediate = S.best.tEnd;
        break;
    case MAPEDimension::Func::VoltageDeviation:
        // Piggy-back on integral, divide it by its length to get mean abs deviation
        if ( S.best.tEnd > 0 )
            factor = 1.0 / S.best.tEnd;
        else
            factor = 1.0 / I.duration;
    case MAPEDimension::Func::VoltageIntegral:
    {
        scalar prevV = I.baseV, prevT = 0., tEnd = S.best.tEnd>0 ? S.best.tEnd : I.duration;
        intermediate = 0.;
        // Calculate I's voltage integral against I.baseV, from t=0 to t=S.best.tEnd (the fitness-relevant bubble's end).
        // Since the stimulation is piecewise linear, decompose it step by step and cumulate the pieces
        for ( const Stimulation::Step &s : I ) {
            scalar sT = s.t, sV = s.V;
            if ( sT > tEnd ) { // Shorten last step to end of observed period
                sT = tEnd;
                if ( s.ramp )
                    sV = (s.V - prevV) * (sT - prevT)/(s.t - prevT);
            }
            if ( s.ramp ) {
                if ( (sV >= I.baseV && prevV >= I.baseV) || (sV <= I.baseV && prevV <= I.baseV) ) { // Ramp does not cross baseV
                    intermediate += factor * (fabs((sV + prevV)/2 - I.baseV) * (sT - prevT));
                } else { // Ramp crosses baseV, treat the two sides separately:
                    scalar r1 = fabs(prevV - I.baseV), r2 = fabs(sV - I.baseV);
                    scalar tCross = r1 / (r1 + r2) * (sT - prevT); //< time from beginning of ramp to baseV crossing
                    intermediate += factor * (r1/2*tCross + r2/2*(sT - prevT - tCross));
                }
            } else {
                intermediate += factor * (fabs(prevV - I.baseV) * (sT - prevT)); // Step only changes to sV at time sT
            }
            prevT = sT;
            prevV = sV;
            if ( s.t >= tEnd )
                break;
        }
        if ( prevT < tEnd ) // Add remainder if last step ends before the observed period - this is never a ramp
            intermediate += factor * (fabs(prevV - I.baseV) * (tEnd - prevT));
    }
        break;
    default:
        intermediate = 0;
        break;
    }

    if ( intermediate < min )
        return 0;
    else if ( intermediate >= max )
        return multiplier * resolution - 1;
    else
        return multiplier * resolution * (intermediate - min)/(max - min);
}

std::string toString(const MAPEDimension::Func &f)
{
    switch ( f ) {
    case MAPEDimension::Func::BestBubbleDuration:   return "BestBubbleDuration";
    case MAPEDimension::Func::BestBubbleTime:       return "BestBubbleTime";
    case MAPEDimension::Func::VoltageDeviation:     return "VoltageDeviation";
    case MAPEDimension::Func::VoltageIntegral:      return "VoltageIntegral";
    }
    return "InvalidFunction";
}

std::ostream &operator<<(std::ostream &os, const MAPEDimension::Func &f)
{
    os << toString(f);
    return os;
}

std::istream &operator>>(std::istream &is, MAPEDimension::Func &f)
{
    std::string s;
    is >> s;
    if ( s == std::string("BestBubbleDuration") )       f = MAPEDimension::Func::BestBubbleDuration;
    else if ( s == std::string("BestBubbleTime") )      f = MAPEDimension::Func::BestBubbleTime;
    else if ( s == std::string("VoltageDeviation") )    f = MAPEDimension::Func::VoltageDeviation;
    else if ( s == std::string("VoltageIntegral") )     f = MAPEDimension::Func::VoltageIntegral;
    else /* Default */ f = MAPEDimension::Func::BestBubbleDuration;
    return is;
}
