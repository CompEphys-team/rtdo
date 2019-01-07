#ifndef WAVEGENSELECTOR_H
#define WAVEGENSELECTOR_H

#include "wavegen.h"

class WavegenSelection : public Result
{
public:
    WavegenSelection(Session &session, size_t archive_idx, Result r = Result());

    Session &session;
    size_t archive_idx;

    struct Range {
        size_t min;
        size_t max;
        bool collapse;
    };
    std::vector<Range> ranges;
    std::vector<bool> paretoMaximise;
    std::vector<size_t> paretoTolerance;
    double paretoFitnessTol = 0;
    bool paretoFront = false;
    double minFitness = 0;
    std::vector<const MAPElite*> selection;

    const Wavegen::Archive &archive() const;
    size_t width(size_t i) const; //!< size of the selected hypercube along the indicated dimension
    size_t size() const; //!< Overall size (i.e. number of selected cells)

    double rmin(size_t i) const; //!< Real-valued min/max for Range i
    double rmax(size_t i) const;

    /**
     * @brief data_relative returns an iterator to the MAPElite at bin coordinates relative to the selected area.
     * @param idx is a vector of bin coordinates corresponding to ranges. Collapsed dimensions' indices are ignored.
     * @param ok is an optional flag that is set to false iff no archive entry exists at the indicated position.
     * @return an iterator into the archive, or to the archive's end() if ok turns false.
     */
    const MAPElite* data_relative(std::vector<size_t> idx, bool *ok = nullptr) const;
    const MAPElite* data_relative(std::vector<double> idx, bool *ok = nullptr) const; //!< Same as above, but behavioural coordinates

    /**
     * @brief data_absolute returns the MAPElite at absolute bin coordinates (i.e. relative to the full bin range of the archive).
     * @param idx is a vector of bin coordinates corresponding to ranges. Collapsed dimensions' indices are ignored.
     * @param ok is an optional flag that is set to false iff either no archive entry exists at the indicated position,
     *  or idx points at a position outside of the selected area.
     * @return an iterator into the archive, or to the archive's end() if ok turns false.
     */
    const MAPElite* data_absolute(std::vector<size_t> idx, bool *ok = nullptr) const;
    const MAPElite* data_absolute(std::vector<double> idx, bool *ok = nullptr) const; //!< Same as above, but behavioural coordinates

    /**
     * @brief limit are short-hand functions to modify ranges. Inputs are forced into the given dimension's maximum range.
     */
    void limit(size_t dimension, double min, double max, bool collapse);
    void limit(size_t dimension, size_t min, size_t max, bool collapse);
    void limit(size_t dimension, Range range);

    /**
     * @brief finalise readies a selection for data queries (@see data_relative(), data_absolute()).
     */
    void finalise();

    QString prettyName() const;

    size_t nFinal; ///!< Total number of valid selected entries. Unlike @fn size(), this does not include empty cells.

    /**
     * @brief getSizeLimit returns the threshold to limit the selection to a given number of valid entries.
     *      Selection must be finalised prior to calling this function.
     * @param n: The desired number of entries
     * @param dimension: The dimension along which the threshold shall be found
     * @param descending: Whether to find the highest (true) or lowest (false) entries along the selected dimension
     * @return the bin value of the threshold
     */
    size_t getSizeLimit(size_t n, size_t dimension, bool descending);

    /**
     * @brief getFitnessSizeLimit returns the minimum fitness to limit the selection to a given number of valid entries.
     *      Selection must be finalised prior to calling this function.
     * @param n: The desired number of entries
     * @return the fitness value of the threshold
     */
    double getFitnessSizeLimit(size_t n);

protected:
    size_t index_relative(const std::vector<size_t> &idx, bool *ok = nullptr) const;
    size_t index_absolute(std::vector<size_t> idx, bool *ok = nullptr) const;

    int dominatesTolerant(const MAPElite *lhs, const MAPElite *rhs) const;
    int dominatesIntolerant(const MAPElite *lhs, const MAPElite *rhs) const;

    void select_uncollapsed();
    void collapse();
    void select_pareto();
};

#endif // WAVEGENSELECTOR_H
