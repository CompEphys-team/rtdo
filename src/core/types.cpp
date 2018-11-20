#include "types.h"
#include <cassert>
#include <cmath>
#include "universallibrary.h"

void Stimulation::insert(Stimulation::Step *position, const Step &value)
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

Stimulation::Stimulation(const iStimulation &I, double dt) :
    duration(I.duration * dt),
    tObsBegin(I.tObsBegin * dt),
    tObsEnd(I.tObsEnd * dt),
    baseV(I.baseV),
    numSteps(I.numSteps)
{
    for ( size_t i = 0; i < numSteps; i++ )
        steps[i] = { scalar(I.steps[i].t * dt),
                     I.steps[i].V,
                     I.steps[i].ramp };
}

void iStimulation::insert(Step *position, const Step &value)
{
    if ( position < steps || position > end() )
        throw std::runtime_error("iStimulation::insert: invalid position.");
    if ( numSteps == Stimulation::maxSteps )
        throw std::runtime_error("iStimulation::insert: Step array is full.");

    for ( Step *it = end(); it > position; it-- ) // Move everything from position to the end right by one
        *it = *(it-1);
    *position = value;
    numSteps++;
    assert(numSteps <= Stimulation::maxSteps);
}

void iStimulation::erase(Step *position)
{
    if ( position < steps || position >= end() )
        throw std::runtime_error("iStimulation::erase: invalid position.");
    for ( Step *it = position; it < end()-1; it++ ) // Move everything from position to the end left by one
        *it = *(it+1);
    numSteps--;
    assert(numSteps > 0);
}

bool iStimulation::operator==(const iStimulation &other) const
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

iStimulation::iStimulation(const Stimulation &I, double dt) :
    duration(lrint(I.duration / dt)),
    tObsBegin(lrint(I.tObsBegin / dt)),
    tObsEnd(lrint(I.tObsEnd / dt)),
    baseV(I.baseV),
    numSteps(I.numSteps)
{
    for ( size_t i = 0; i < numSteps; i++ )
        steps[i] = { int(lrint(I.steps[i].t / dt)),
                     I.steps[i].V,
                     I.steps[i].ramp };
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

bool iStimulation::Step::operator==(const iStimulation::Step &other) const
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
    case IntegrationMethod::RungeKuttaFehlberg45: os << "RungeKuttaFehlberg45"; break;
    }
    return os;
}
std::istream& operator>>(std::istream& is, IntegrationMethod &m)
{
    std::string s;
    is >> s;
    if ( s == std::string("ForwardEuler") )     m = IntegrationMethod::ForwardEuler;
    else if ( s == std::string("RungeKutta4") ) m = IntegrationMethod::RungeKutta4;
    else if ( s == std::string("RungeKuttaFehlberg45") ) m = IntegrationMethod::RungeKuttaFehlberg45;
    else /* Default: */ m = IntegrationMethod::RungeKutta4;
    return is;
}

bool MAPElite::compete(const MAPElite &rhs)
{
    if ( rhs.fitness > fitness ) {
        fitness = rhs.fitness;
        wave = rhs.wave;
        obs = rhs.obs;
        return true;
    }
    return false;
}

size_t MAPEDimension::bin(const iStimulation &I, size_t multiplier, double dt) const
{
    scalar intermediate = 0.0;
    scalar factor = dt;
    switch ( func ) {
    case MAPEDimension::Func::BestBubbleDuration:
        intermediate = (I.tObsEnd - I.tObsBegin) * dt;
        break;
    case MAPEDimension::Func::BestBubbleTime:
        intermediate = I.tObsBegin * dt;
        break;
    case MAPEDimension::Func::VoltageDeviation:
        // Piggy-back on integral, divide it by its length to get mean abs deviation
        if ( I.tObsEnd > 0 )
            factor = 1./I.tObsEnd;
        else
            factor = 1./I.duration;
    case MAPEDimension::Func::VoltageIntegral:
    {
        scalar prevV = I.baseV;
        int prevT = 0, tEnd = I.tObsEnd>0 ? I.tObsEnd : I.duration;
        intermediate = 0.;
        // Calculate I's voltage integral against I.baseV, from t=0 to t=S.best.tEnd (the fitness-relevant bubble's end).
        // Since the stimulation is piecewise linear, decompose it step by step and cumulate the pieces
        for ( const iStimulation::Step &s : I ) {
            int sT = s.t;
            scalar sV = s.V;
            if ( sT > tEnd ) { // Shorten last step to end of observed period
                sT = tEnd;
                if ( s.ramp )
                    sV = ((s.V - prevV) * (sT - prevT))/(s.t - prevT);
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

    return bin(intermediate, multiplier);
}

size_t MAPEDimension::bin(scalar value, size_t multiplier) const
{
    if ( value < min )
        return 0;
    else if ( value >= max )
        return multiplier * resolution - 1;
    else
        return multiplier * resolution * (value - min)/(max - min);
}

size_t MAPEDimension::bin(const iStimulation &I, size_t paramIdx, size_t clusterIdx, size_t nClusters, size_t multiplier, double dt) const
{
    switch ( func ) {
    case Func::EE_ClusterIndex:
        return clusterIdx;
    case Func::EE_NumClusters:
        return nClusters;
    case Func::EE_ParamIndex:
        return paramIdx;
    default:
        return bin(I, multiplier, dt);
    }
}

scalar MAPEDimension::bin_inverse(size_t bin, size_t multiplier) const
{
    switch ( func ) {
    case Func::EE_ClusterIndex:
    case Func::EE_NumClusters:
    case Func::EE_ParamIndex:
        return bin;
    default:
        return min + bin * (max - min)/(multiplier * resolution);
    }
}

void MAPEDimension::setDefaultMinMax(StimulationData d, size_t nParams)
{
    scalar maxDeviation = d.maxVoltage-d.baseV > d.baseV-d.minVoltage
            ? d.maxVoltage - d.baseV
            : d.baseV - d.minVoltage;
    switch ( func ) {
    case Func::BestBubbleDuration: min = 0; max = d.duration; return;
    case Func::BestBubbleTime:     min = 0; max = d.duration; return;
    case Func::VoltageDeviation:   min = 0; max = maxDeviation; return;
    case Func::VoltageIntegral:    min = 0; max = maxDeviation * d.duration; return;
    case Func::EE_ClusterIndex:
    case Func::EE_NumClusters:     min = 0; max = UniversalLibrary::maxClusters; return;
    case Func::EE_ParamIndex:      min = 0; max = nParams; return;
    }
}

std::string toString(const MAPEDimension::Func &f)
{
    switch ( f ) {
    case MAPEDimension::Func::BestBubbleDuration:   return "BestBubbleDuration";
    case MAPEDimension::Func::BestBubbleTime:       return "BestBubbleTime";
    case MAPEDimension::Func::VoltageDeviation:     return "VoltageDeviation";
    case MAPEDimension::Func::VoltageIntegral:      return "VoltageIntegral";
    case MAPEDimension::Func::EE_ClusterIndex:      return "ClusterIndex";
    case MAPEDimension::Func::EE_NumClusters:       return "NumClusters";
    case MAPEDimension::Func::EE_ParamIndex:        return "ParamIndex";
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
    else if ( s == std::string("ClusterIndex") )        f = MAPEDimension::Func::EE_ClusterIndex;
    else if ( s == std::string("NumClusters") )         f = MAPEDimension::Func::EE_NumClusters;
    else if ( s == std::string("ParamIndex") )          f = MAPEDimension::Func::EE_ParamIndex;
    else /* Default */ f = MAPEDimension::Func::BestBubbleDuration;
    return is;
}

std::string toString(const FilterMethod &m)
{
    switch ( m ) {
    case FilterMethod::MovingAverage:   return "MovingAverage";
    case FilterMethod::SavitzkyGolay23: return "SavitzkyGolay23";
    case FilterMethod::SavitzkyGolay45: return "SavitzkyGolay45";
    case FilterMethod::SavitzkyGolayEdge3: return "SavitzkyGolayEdge3";
    case FilterMethod::SavitzkyGolayEdge5: return "SavitzkyGolayEdge5";
    }
    return "InvalidMethod";
}

std::ostream &operator<<(std::ostream &os, const FilterMethod &m)
{
    os << toString(m);
    return os;
}

std::istream &operator>>(std::istream &is, FilterMethod &m)
{
    std::string s;
    is >> s;
    if ( s == std::string("MovingAverage") )        m = FilterMethod::MovingAverage;
    else if ( s == std::string("SavitzkyGolay23") ) m = FilterMethod::SavitzkyGolay23;
    else if ( s == std::string("SavitzkyGolay45") ) m = FilterMethod::SavitzkyGolay45;
    else /* Default */ m = FilterMethod::MovingAverage;
    return is;
}
