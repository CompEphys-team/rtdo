#ifndef ERRORPROFILER_H
#define ERRORPROFILER_H

#include "sessionworker.h"
#include "experimentlibrary.h"

class ErrorProfile
{
public:
    class Profile;
    class Iterator;

    ErrorProfile(Session &session);

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

    void setStimulations(std::vector<Stimulation> &&stim); //!< Sets the stimulations to be considered
    const inline std::vector<Stimulation> &stimulations() const { return m_stimulations; }

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
    ExperimentLibrary &lib;
    std::vector<Permutation> m_permutations; //!< Input: Describes how each parameter is perturbed
    std::vector<Stimulation> m_stimulations; //!< Input: The waveforms under consideration
    std::list<std::vector<scalar>> errors; //!< Raw output

    /// Profiling workhorse, called through ErrorProfiler::generate
    void generate(const Stimulation &stim, std::vector<scalar> &errors, DAQ *daq, scalar settleDuration);

    /// Profiling helpers
    void settle(scalar baseV, DAQ *daq, scalar settleDuration);
    void stimulate(const Stimulation &stim, DAQ *daq);

    friend class ErrorProfiler;
};


/**
 * @brief The ErrorProfiler class is a tool for grid sampling style sensitivity analysis.
 *
 * In a nutshell, it can vary one or more parameters of a model within a set range,
 * stimulate the perturbed models, and record the error that results. The results
 * are offered as a set of vector-like objects (@see ErrorProfile::Profile), each
 * containing the values seen along one axis of the sampling grid.
 * The idea is to get a sense of how sensitive a certain stimulation is to changes
 * in its target parameter over a wide range of candidate models.
 */
class ErrorProfiler : public SessionWorker
{
    Q_OBJECT

public:
    ErrorProfiler(Session &session, DAQ *daq = nullptr);
    ~ErrorProfiler();

    void abort(); //!< Abort all queued slot actions.

    /**
     * @brief queueProfile adds a new ErrorProfile to the queue for generate() to work with.
     * A typical workflow will not require multiple ErrorProfiles in the queue, but passing them
     * directly to the slot is tricky. To clear the queue, call abort().
     */
    inline void queueProfile(ErrorProfile &&p) { m_queue.push_back(p); }

    const inline std::vector<ErrorProfile> &profiles() const { return m_profiles; }

public slots:
    /**
     * @brief generate runs the simulations set up in the first queued ErrorProfile.
     * A progress() signal is emitted at the end of each simulation block to allow users to keep track of progress.
     * When the profiling is complete, a done() signal is also emitted. Only then does the resulting ErrorProfile
     * become accessible through profiles().
     */
    void generate();

signals:
    void progress(int nth, int total);
    void done();
    void didAbort();

protected slots:
    void clearAbort();

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "ErrorProfiler"; }

private:
    DAQ *simulator;
    DAQ *daq;

    bool aborted;

    std::list<ErrorProfile> m_queue;
    std::vector<ErrorProfile> m_profiles;
};

#endif // ERRORPROFILER_H
