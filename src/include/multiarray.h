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


#ifndef MULTIARRAY_H
#define MULTIARRAY_H

#include <vector>
#include <cassert>

using std::size_t;

template<typename T> class MultiArray
{
public:
    MultiArray(const std::vector<std::size_t> &dims) :
        m_dims(dims),
        m_data(product(m_dims))
    {}

    MultiArray(const std::vector<size_t>& dims, const T& initial) :
        m_dims(dims),
        m_data(product(m_dims), initial)
    {}

    inline const std::vector<int>& dims() const { return m_dims; }

    inline size_t size() const { return m_data.size(); }

    inline const T& operator[](const std::vector<size_t>& indices) const {
        return m_data.at(index(indices));
    }

    inline T& operator[](const std::vector<size_t>& indices) {
        return m_data[index(indices)];
    }

    inline const T& operator[](size_t idx) const {
        assert(idx < m_data.size());
        return m_data.at(idx);
    }

    inline T& operator[](size_t idx) {
        assert(idx < m_data.size());
        return m_data[idx];
    }

private:
    std::vector<size_t> m_dims;
    std::vector<T> m_data;

    static size_t product(const std::vector<size_t>& dims) {
        size_t result = 1;
        for ( size_t i : dims )
            result *= i;
        return result;
    }

    size_t index(const std::vector<size_t>& indices) const {
        size_t v = 0;
        for (size_t i = 0; i<m_dims.size(); ++i) {
            assert(indices[i] < m_dims[i]);
            if (i) v *= m_dims[i];
            v += indices[i];
        }
        return v;
    }
};

#endif // MULTIARRAY_H
