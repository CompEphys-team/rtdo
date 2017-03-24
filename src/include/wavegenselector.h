#ifndef WAVEGENSELECTOR_H
#define WAVEGENSELECTOR_H

#include <QObject>
#include <QFile>
#include "wavegen.h"

class WavegenSelection
{
private:
    Session &session;
    size_t archive_idx;

    WavegenSelection(Session &session, size_t archive_idx);
    friend class WavegenSelector;

public:
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
};

class WavegenSelector : public QObject
{
    Q_OBJECT
public:
    WavegenSelector(Session &session);

    Session &session;

    WavegenSelection select(size_t wavegen_archive_idx) const;
    void limit(WavegenSelection &selection, size_t dimension, double min, double max, bool collapse) const;
    void limit(WavegenSelection &selection, size_t dimension, size_t min, size_t max, bool collapse) const;
    void limit(WavegenSelection &selection, size_t dimension, WavegenSelection::Range range) const;
    void finalise(WavegenSelection &selection) const;

    void save(WavegenSelection &selection); //!< Finalises selection and adds it to the database.

    inline const std::vector<WavegenSelection> &selections() const { return m_selections; }

    void load(const QString &action, const QString &args, QFile &results);

protected:
    std::vector<WavegenSelection> m_selections;

    const static QString action;
    const static quint32 magic, version;

protected slots:
    void log(int idx);

signals:
    void saved(int idx, QPrivateSignal);
};

#endif // WAVEGENSELECTOR_H
