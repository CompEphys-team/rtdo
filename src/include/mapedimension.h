#ifndef MAPEDIMENSION_H
#define MAPEDIMENSION_H

#include "types.h"

/**
 * @brief The MAPEDimension class is a prototype for all MAP-Elites dimensions. A dimension must define its size,
 * i.e. the total number of bins along its axis, and be able to classify a given Stimulation/WaveStats combination
 * into one of those bins by way of specifying a behavioural measure. Some useful examples are given below.
 */
class MAPEDimension
{
public:
    MAPEDimension() {}
    virtual ~MAPEDimension() {}

    /**
     * @brief size indicates the total number of bins along this dimension.
     */
    virtual size_t size() const = 0;

    /**
     * @brief bin calculates the bin which the given Stimulation with its performance statistics belongs to.
     * @param I is a stimulation that has been evaluated
     * @param S is the performance statistics blob for @param I
     * @return a coordinate along this dimension
     */
    inline size_t bin(const Stimulation &I, const WaveStats &S) const { return bin(behaviour(I, S)); }

    /**
     * @brief bin calculates the bin which a given behaviour should fall into.
     * @param behaviour is a behavioural measure, e.g. an average across several evaluations
     * @return a coordinate along this dimension
     */
    virtual size_t bin(double behaviour) const = 0;

    /**
     * @brief behaviour calculates a single numerical value as a behavioural measure of the given Stimulation,
     * based on its specification and/or performance statistics.
     * @param I is the stimulation that has been evaluated
     * @param S is the performance statistics blob for @param I
     * @return a normalised, but not discretised, behaviour value in this dimension
     */
    virtual double behaviour(const Stimulation &I, const WaveStats &S) const = 0;

protected:
    inline size_t delimit(double value) const
    {
        if ( value < 0 )
            return 0;
        if ( value > size()-1 )
            return size()-1;
        return size_t(round(value));
    }
};

//!< Number of waveform steps
class MAPED_numSteps : public MAPEDimension
{
public:
    MAPED_numSteps(const StimulationData &p) :
        _size(p.maxSteps - p.minSteps + 1),
        _offset(p.minSteps)
    {}
    ~MAPED_numSteps() {}
    inline size_t size() const { return _size; }
    inline size_t bin(double behaviour) const { return delimit(behaviour - _offset); }
    inline double behaviour(const Stimulation &I, const WaveStats &) const { return I.size(); }
private:
    size_t _size, _offset;
};

//!< Number of Buds/Bubbles
class MAPED_numB : public MAPEDimension
{
public:
    MAPED_numB(const StimulationData &p, int WaveStats::* n) :
        _size(2 * p.maxSteps),
        n(n)
    {}
    ~MAPED_numB() {}
    inline size_t size() const { return _size; }
    inline size_t bin(double behaviour) const { return delimit(behaviour); }
    inline double behaviour(const Stimulation &, const WaveStats &S) const { return S.*n; }
private:
    size_t _size;
    int WaveStats::* n;
};

//!< Scalar member of a bud/bubble
class MAPED_BScalar : public MAPEDimension
{
public:
    MAPED_BScalar(WaveStats::Bubble WaveStats::* B, scalar WaveStats::Bubble::* var, scalar min, scalar max, size_t size) :
        _size(size),
        min(min),
        max(max),
        B(B),
        var(var)
    {}
    ~MAPED_BScalar() {}
    inline size_t size() const { return _size; }
    inline size_t bin(double behaviour) const { return delimit((behaviour - min) / (max - min) * (_size - 1)); }
    inline double behaviour(const Stimulation &, const WaveStats &S) const { return S.*B.*var; }
private:
    size_t _size;
    scalar min, max;
    WaveStats::Bubble WaveStats::* B;
    scalar WaveStats::Bubble::* var;
};

//!< Duration of a bud/bubble
class MAPED_BCycles : public MAPEDimension
{
public:
    MAPED_BCycles(WaveStats::Bubble WaveStats::* B, size_t maxCycles, size_t size) :
        _size(size),
        max(maxCycles),
        B(B)
    {}
    ~MAPED_BCycles() {}
    inline size_t size() const { return _size; }
    inline size_t bin(double behaviour) const { return delimit(behaviour / max * (_size - 1)); }
    inline double behaviour(const Stimulation &, const WaveStats &S) const { return (S.*B).cycles; }
private:
    size_t _size, max;
    WaveStats::Bubble WaveStats::* B;
};

#endif // MAPEDIMENSION_H
