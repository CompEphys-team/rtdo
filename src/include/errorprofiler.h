#ifndef ERRORPROFILER_H
#define ERRORPROFILER_H

#include <QObject>
#include "experimentlibrary.h"

/**
 * @brief The ErrorProfiler class is a tool for grid sampling style sensitivity analysis.
 *
 * In a nutshell, it can vary one or more parameters of a model within a set range,
 * stimulate the perturbed models, and record the error that results. The results
 * are offered as a set of vector-like objects (@see ErrorProfiler::Profile), each
 * containing the values seen along one axis of the sampling grid.
 * The idea is to get a sense of how sensitive a certain stimulation is to changes
 * in its target parameter over a wide range of candidate models.
 */
class ErrorProfiler : public QObject
{
    Q_OBJECT

public:
    ErrorProfiler(ExperimentLibrary &lib, DAQ *daq = nullptr);
    ~ErrorProfiler();

    ExperimentLibrary &lib;

    /**
     * @brief The Permutation struct is ErrorProfiler's governing data type.
     */
    struct Permutation {
        Permutation() : n(1), min(0), max(0), fixed(false) {}

        /**
         * @brief n is the number of parameter values to be used.
         * If n is 0 (default), the number is set to lib.expd.numCandidates.
         * If n is 1, only the parameter's initial value is used.
         */
        size_t n;

        /**
         * @brief min and max are the lower and upper bound of the value range used.
         * Both values are included in the distribution of parameter values.
         * If min == max == 0 (default), the range specified in lib.adjustableParams is used instead.
         */
        double min, max;

        /**
         * @brief fixed is a flag that, when set to true, forces the parameter to take only a single value (@see value).
         */
        bool fixed;

        /**
         * @brief value is the sole parameter value used if the fixed flag is set (@see fixed).
         */
        double value;
    };

    class Iterator
    {
    public:
        Iterator(std::vector<scalar>::const_iterator origin, size_t stride, size_t idx) : origin(origin), stride(stride), idx(idx) {}
        inline const scalar &operator *() { return *(origin + stride*idx); }
        inline Iterator &operator ++() { ++idx; return *this; }
        inline Iterator &operator +=(size_t offset) { idx += offset; return *this; }
        inline bool operator !=(const Iterator &rhs) { return idx != rhs.idx; }
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
        inline const scalar &operator [](size_t idx) { return *(origin + stride*idx); }
        inline Iterator begin() { return Iterator(origin, stride, 0); }
        inline Iterator end() { return Iterator(origin, stride, _size); }
        inline size_t size() { return _size; }
        inline std::vector<size_t> paramIndices() { return pIdx; }
        inline size_t paramIndex(size_t i) { return pIdx[i]; }
    private:
        const std::vector<scalar>::const_iterator origin;
        const size_t stride, _size;
        std::vector<size_t> pIdx;
    };

    /**
     * @brief setPermutations sets the data governing the ErrorProfiler, discarding previous settings and profiles.
     * Note, this invalidates any previously returned Profiles.
     * @param p is a vector of Permutation objects, each corresponding to the adjustable parameter of the same index.
     */
    void setPermutations(std::vector<Permutation> p);

    size_t getNumPermutations(); //!< @brief getNumPermutations returns the total number of models run under current settings
    size_t getNumSimulations(); //!< @brief getNumSimulations returns the number of simulation runs required to profile

    void setStimulations(std::vector<Stimulation> stims); //!< @brief Sets a number of stimulations to be profiled

    /**
     * @brief getParameterValue gives a target parameter's value at a given position in its range.
     * The values are uniformly distributed across the full range of the parameter and include the lower and upper bounds.
     * Indices can be taken e.g. from Iterator::index() or from Profile::paramIndices().
     */
    double getParameterValue(size_t param, size_t idx);

    /**
     * @brief getParameterIndex gives the index in a target parameter's distribution that most closely matches the given value.
     * Use this to find the profile of a specific model. Within resolution limits, x == getParameterValue(t, getParameterIndex(t, x)).
     */
    size_t getParameterIndex(size_t param, double value);

    /**
     * @brief getProfiles produces a set of iterable error profiles from the perspective of a given target parameter.
     * Each Profile contains the error values for the full range of the target parameter, while all other parameters
     * are held constant.
     */
    std::vector<ErrorProfiler::Profile> getProfiles(size_t targetParam, const std::vector<scalar> &profile);

    void abort(); //!< Abort all queued slot actions.

    std::list<std::vector<scalar>> profiles;

public slots:
    /**
     * @brief profile runs the requested simulations for each of the stimulations set by setStimulations().
     * Results are deposited in ErrorProfiler::profiles. A profileComplete(size_t i) signal is emitted at the end
     * of each simulation; all profiles from index 0 to i can be safely accessed.
     */
    void profile();

signals:
    void profileComplete(int index);
    void done();
    void didAbort();

protected slots:
    void clearAbort();

private:
    DAQ *simulator;
    DAQ *daq;

    std::vector<Permutation> permutations;
    std::vector<Stimulation> stimulations;

    std::vector<scalar> errors;

    bool aborted;

    void settle(scalar baseV);
    void stimulate(const Stimulation &stim);

    /**
     * @brief profile runs the requested simulations, populating the errors vector.
     */
    void profile(const Stimulation &stim);
};

#endif // ERRORPROFILER_H
