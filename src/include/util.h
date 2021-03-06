/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#ifndef UTIL_H
#define UTIL_H

#include <cmath>
#include <string>
#include <iostream>
#include <algorithm>
#include <vector>

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

/// A string that saves/loads with quotes. Treat it like a std::string, but get quoted AP values.
/// For backwards compatibility, operator>> will read from the first non-whitespace character it finds;
///  if that isn't a double quote ("), it'll read up to the next whitespace as if it were a normal string read.
class QuotedString : public std::string {
public:
    template <typename... Args> QuotedString(Args... args) : std::string(args...) {}
};
std::istream &operator>>(std::istream &is, QuotedString &str);
std::ostream &operator<<(std::ostream &os, const QuotedString &str);


// Interpolated quartiles -- https://stackoverflow.com/a/37708864
template<typename T>
inline double Lerp(T v0, T v1, T t)
{
    return (1 - t)*v0 + t*v1;
}

template<typename T>
inline std::vector<T> Quantile(const std::vector<T>& inData, const std::vector<T>& probs)
{
    if (inData.empty())
    {
        return std::vector<T>();
    }

    if (1 == inData.size())
    {
        return std::vector<T>(probs.size(), inData[0]);
    }

    std::vector<T> data = inData;
    std::sort(data.begin(), data.end());
    std::vector<T> quantiles;

    for (size_t i = 0; i < probs.size(); ++i)
    {
        T poi = Lerp<T>(-0.5, data.size() - 0.5, probs[i]);

        size_t left = std::max(int64_t(std::floor(poi)), int64_t(0));
        size_t right = std::min(int64_t(std::ceil(poi)), int64_t(data.size() - 1));

        T datLeft = data.at(left);
        T datRight = data.at(right);

        T quantile = Lerp<T>(datLeft, datRight, poi - left);

        quantiles.push_back(quantile);
    }

    return quantiles;
}

#endif // UTIL_H
