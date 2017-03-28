#ifndef WAVEGENSELECTOR_H
#define WAVEGENSELECTOR_H

#include "wavegen.h"

class WavegenSelection
{
public:
    WavegenSelection(Session &session, size_t archive_idx);

    Session &session;
    size_t archive_idx;

    struct Range {
        size_t min;
        size_t max;
        bool collapse;
    };
    std::vector<Range> ranges;
    std::vector<std::list<MAPElite>::const_iterator> selection;

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
    std::list<MAPElite>::const_iterator data_relative(std::vector<size_t> idx, bool *ok = nullptr) const;
    std::list<MAPElite>::const_iterator data_relative(std::vector<double> idx, bool *ok = nullptr) const; //!< Same as above, but behavioural coordinates

    /**
     * @brief data_absolute returns the MAPElite at absolute bin coordinates (i.e. relative to the full bin range of the archive).
     * @param idx is a vector of bin coordinates corresponding to ranges. Collapsed dimensions' indices are ignored.
     * @param ok is an optional flag that is set to false iff either no archive entry exists at the indicated position,
     *  or idx points at a position outside of the selected area.
     * @return an iterator into the archive, or to the archive's end() if ok turns false.
     */
    std::list<MAPElite>::const_iterator data_absolute(std::vector<size_t> idx, bool *ok = nullptr) const;
    std::list<MAPElite>::const_iterator data_absolute(std::vector<double> idx, bool *ok = nullptr) const; //!< Same as above, but behavioural coordinates

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
};

class WavegenSelector : public SessionWorker
{
    Q_OBJECT
public:
    WavegenSelector(Session &session) : SessionWorker(session) {}

    void save(const WavegenSelection &selection); //!< Finalises selection and adds it to the database.

    inline const std::vector<WavegenSelection> &selections() const { return m_selections; }

    QString prettyName(int n) const; //!< Returns a string describing the nth WavegenSelection

protected:
    friend class Session;
    void load(const QString &action, const QString &args, QFile &results);
    inline QString actorName() const { return "WavegenSelector"; }

    std::vector<WavegenSelection> m_selections;

    const static QString action;
    const static quint32 magic, version;
};

#endif // WAVEGENSELECTOR_H