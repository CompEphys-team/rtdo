#ifndef UTIL_H
#define UTIL_H

#include <cmath>
using std::size_t;

//! Linear projection of idx from [0, size-1] into [min, max]
inline double linSpace(double min, double max, size_t size, size_t idx)
{
    return min + (max-min) * idx / (size-1);
}

inline size_t linSpaceInverse(double min, double max, size_t size, double value)
{
    double i = (value-min) / (max-min) * (size-1);
    return i > 0 ? std::round(i) : 0;
}

//! Logarithmic projection of idx from [0, size-1] into [min, max]
inline double logSpace(double min, double max, size_t size, size_t idx)
{
    // Band-aid for 0-bounded intervals
    if ( min == 0 ) min = max / std::pow(size-1, 2);
    if ( max == 0 ) max = min / std::pow(size-1, 2);
    // Distribute x linear-uniformly in [log(min), log(max)] : x = idx/(size-1) * (log(max)-log(min)) + log(min)
    // Then exponentiate to get a log-distributed value y in [min,max] : y = e^x
    return min * std::pow(max/min, double(idx)/(size-1));
}

inline size_t logSpaceInverse(double min, double max, size_t size, double value)
{
    if ( min == 0 ) min = max / std::pow(size-1, 2);
    if ( max == 0 ) max = min / std::pow(size-1, 2);
    double i = (size-1) * std::log(value/min) / std::log(max/min);
    return i > 0 ? std::round(i) : 0;
}

#endif // UTIL_H
