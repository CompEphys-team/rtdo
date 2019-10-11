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


#ifndef ERRORPROFILE_H
#define ERRORPROFILE_H

#include "types.h"
#include "wavesource.h"
#include "universallibrary.h"
#include "profilestats.h"

class ErrorProfile : public Result
{
public:
    class Profile;
    class Iterator;

    ErrorProfile(Session &session, Result r = Result());

    /**
     * @brief The Permutation struct is ErrorProfile's governing data type.
     */
    struct Permutation {
        Permutation() : n(1), min(0), max(0), fixed(false) {}

        /**
         * @brief n is the number of parameter values to be used.
         * If n is 0, the number is set to Project::expNumCandidates().
         * If n is 1 (default), only the parameter's initial value is used.
         * To get a single value other than the initial, use fixed and value.
         */
        size_t n;

        /**
         * @brief min and max are the lower and upper bound of the value range used.
         * Both values are included in the distribution of parameter values.
         * If min == max == 0 (default), the range specified in Project::model().adjustableParams is used instead.
         */
        double min, max;

        bool fixed; //!< fixed is a flag that, when set to true, forces the parameter to take only a single value (@see value).
        double value; //!< value is the sole parameter value used if the fixed flag is set (@see fixed).
    };

    void setPermutations(std::vector<Permutation> p); //!< Replace the full set of permutations, ordered by parameter
    void setPermutation(size_t param, Permutation p); //!< Replaces the existing permutation for parameter param
    const inline std::vector<Permutation> &permutations() const { return m_permutations; }

    void setSource(WaveSource src); //!< Sets the stimulations from an existing source
    const inline WaveSource &source() const { return m_src; }
    const inline std::vector<iStimulation> &stimulations() const { return m_stimulations; }
    const inline std::vector<iObservations> &observations() const { return m_observations; }

    size_t numPermutations() const; //!< @brief numPermutations returns the total number of models run under current settings
    size_t numSimulations() const; //!< @brief numSimulations returns the number of simulation runs required to profile

    /**
     * @brief parameterValue gives a target parameter's value at a given position in its range.
     * The values are uniformly distributed across the full range of the parameter and include the lower and upper bounds.
     * Indices can be taken e.g. from Profile::paramIndices() or Iterator::index().
     */
    double parameterValue(size_t param, size_t idx) const;
    std::vector<double> parameterValues(size_t param, std::vector<size_t> idx) const; //!< Shorthand for parameterValue()

    /**
     * @brief parameterIndex gives the index in a target parameter's distribution that most closely matches the given value.
     * Use this to find the profile of a specific model. Within resolution limits, x == parameterValue(t, parameterIndex(t, x)).
     */
    size_t parameterIndex(size_t param, double value) const;

    /**
     * @brief profiles produces a set of iterable error profiles from the perspective of a given target parameter.
     * Each Profile contains the error values for the full range of the target parameter, while all other parameters
     * are held constant. To find this constant value, use e.g. parameterValue(param, Profile::paramIndex(param)).
     * Each inner vector<Profile> contains all profiles thus constructed for one Stimulation. The order of inner vectors
     * corresponds to the order of stimulations().
     */
    std::vector<std::vector<Profile>> profiles(size_t targetParam) const;

    /**
     * @brief stats returns a statistics object for the given target parameter.
     */
    inline const ProfileStats &stats(size_t targetParam) const { return m_stats[targetParam]; }

    QString prettyName() const;

    class Iterator
    {
    public:
        Iterator(std::vector<scalar>::const_iterator origin, size_t stride, size_t idx) : origin(origin), stride(stride), idx(idx) {}
        inline const scalar &operator *() const { return *(origin + stride*idx); }
        inline Iterator &operator ++() { ++idx; return *this; }
        inline Iterator &operator +=(size_t offset) { idx += offset; return *this; }
        inline bool operator !=(const Iterator &rhs) const { return idx != rhs.idx; }
        inline size_t index() const { return idx; }
    private:
        const std::vector<scalar>::const_iterator origin;
        const size_t stride;
        size_t idx;
    };

    class Profile
    {
    public:
        Profile(std::vector<scalar>::const_iterator origin, size_t stride, size_t size, std::vector<size_t> pIdx) :
            origin(origin), stride(stride), _size(size), pIdx(pIdx) {}
        inline const scalar &operator [](size_t idx) const { return *(origin + stride*idx); }
        inline Iterator begin() const { return Iterator(origin, stride, 0); }
        inline Iterator end() const { return Iterator(origin, stride, _size); }
        inline size_t size() const { return _size; }
        inline std::vector<size_t> paramIndices() const { return pIdx; }
        inline size_t paramIndex(size_t i) const { return pIdx[i]; }
    private:
        const std::vector<scalar>::const_iterator origin;
        const size_t stride, _size;
        std::vector<size_t> pIdx;
    };

private:
    UniversalLibrary &lib;
    std::vector<Permutation> m_permutations; //!< Input: Describes how each parameter is perturbed
    std::vector<iStimulation> m_stimulations; //!< Derived input: The waveforms under consideration, provided via m_src
    std::vector<iObservations> m_observations;
    WaveSource m_src; //!< Input: waveform source
    std::list<std::vector<scalar>> errors; //!< Raw output
    std::vector<ProfileStats> m_stats;

    /// Profiling workhorse, called through ErrorProfiler::generate
    void generate(const iStimulation &stim, const iObservations &obs, std::vector<scalar> &errors);

    /// Stats processing
    void process_stats();

    /// Save/load
    friend QDataStream &operator<<(QDataStream &os, const ErrorProfile &);
    friend QDataStream &operator>>(QDataStream &is, ErrorProfile &);

    friend class ErrorProfiler;

    int version; //!< Used during loading only
    Session &session; //!< Used during loading only
};

#endif // ERRORPROFILE_H
