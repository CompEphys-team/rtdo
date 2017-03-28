#include "wavegenselector.h"
#include "session.h"

WavegenSelection::WavegenSelection(Session &session, size_t archive_idx) :
    session(session),
    archive_idx(archive_idx),
    ranges(archive().searchd.mapeDimensions.size())
{
}

const Wavegen::Archive &WavegenSelection::archive() const
{
    return session.wavegen().archives().at(archive_idx);
}

size_t WavegenSelection::width(size_t i) const
{
    if ( ranges[i].collapse )
        return 1;
    else
        return ranges[i].max - ranges[i].min + 1;
}

size_t WavegenSelection::size() const
{
    size_t ret = 1;
    for ( size_t i = 0; i < ranges.size(); i++ )
        ret *= width(i);
    return ret;
}

std::list<MAPElite>::const_iterator WavegenSelection::data_relative(std::vector<size_t> idx, bool *ok) const
{
    size_t index = 0, multiplier = 1;
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( !ranges[i].collapse && idx[i] >= width(i) ) {
            if ( ok )
                *ok = false;
            return archive().elites.end();
        }
        index += idx[i]*multiplier;
        multiplier *= width(i);
    }
    auto ret = selection.at(index);
    if ( ok )
        *ok = (ret != archive().elites.end());
    return ret;
}

std::list<MAPElite>::const_iterator WavegenSelection::data_relative(std::vector<double> idx, bool *ok) const
{
    std::vector<size_t> bin(ranges.size());
    size_t multiplier = session.wavegen().mape_multiplier(archive().precision);
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( ranges[i].collapse )
            continue;
        scalar min = archive().searchd.mapeDimensions[i].bin_inverse( ranges[i].min, multiplier ); // Behaviour-value offset of the selected area
        scalar shifted = min + idx[i]; // Absolute behavioural value
        bin[i] = archive().searchd.mapeDimensions[i].bin(shifted, multiplier); // guaranteed valid bin for whole archive, but not for selection
    }
    return data_absolute(bin, ok); // Let data_absolute deal with out of range issues
}

std::list<MAPElite>::const_iterator WavegenSelection::data_absolute(std::vector<size_t> idx, bool *ok) const
{
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( ranges[i].collapse ) {
            // ignore collapsed dimension
            idx[i] = 0;
        } else if ( idx[i] < ranges[i].min || idx[i] > ranges[i].max ) {
            // idx out of selected range, fail
            if ( ok )
                *ok = false;
            return archive().elites.end();
        } else {
            // Offset index to range-relative
            idx[i] -= ranges[i].min;
        }
    }
    return data_relative(idx, ok);
}

std::list<MAPElite>::const_iterator WavegenSelection::data_absolute(std::vector<double> idx, bool *ok) const
{
    std::vector<size_t> bin(ranges.size());
    size_t multiplier = session.wavegen().mape_multiplier(archive().precision);
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        bin[i] = archive().searchd.mapeDimensions[i].bin(idx[i], multiplier);
    }
    return data_absolute(bin, ok);
}





WavegenSelector::WavegenSelector(Session &session) :
    SessionWorker(session)
{
}

WavegenSelection WavegenSelector::select(size_t wavegen_archive_idx) const
{
    return WavegenSelection(session, wavegen_archive_idx);
}

void WavegenSelector::limit(WavegenSelection &selection, size_t dimension, double min, double max, bool collapse) const
{
    const MAPEDimension &dim = selection.archive().searchd.mapeDimensions.at(dimension);
    size_t multiplier = session.wavegen().mape_multiplier(selection.archive().precision);
    limit(selection, dimension, WavegenSelection::Range{dim.bin(min, multiplier), dim.bin(max, multiplier), collapse});
}

void WavegenSelector::limit(WavegenSelection &selection, size_t dimension, size_t min, size_t max, bool collapse) const
{
    limit(selection, dimension, WavegenSelection::Range{min, max, collapse});
}

void WavegenSelector::limit(WavegenSelection &selection, size_t dimension, WavegenSelection::Range range) const
{
    const MAPEDimension &dim = selection.archive().searchd.mapeDimensions.at(dimension);
    size_t multiplier = session.wavegen().mape_multiplier(selection.archive().precision);
    size_t rmax = multiplier * dim.resolution - 1;
    if ( range.max < range.min )
        range.max = range.min;
    if ( range.max > rmax )
        range.max = rmax;
    if ( range.min > rmax )
        range.min = rmax;
    selection.ranges.at(dimension) = range;
}

void WavegenSelector::finalise(WavegenSelection &selection) const
{
    const size_t dimensions = selection.ranges.size();
    std::list<MAPElite>::const_iterator default_iterator = selection.archive().elites.end();
    size_t uncollapsed_size = 1, collapsed_size = 1;
    std::vector<size_t> offsets, sizes;
    std::vector<size_t> offset_index(dimensions, 0);
    std::vector<size_t> true_index(dimensions);
    for ( WavegenSelection::Range const& r : selection.ranges ) {
        size_t s(r.max - r.min + 1);
        uncollapsed_size *= s;
        collapsed_size *= r.collapse ? 1 : s;
        offsets.push_back(r.min);
        sizes.push_back(s);
    }
    std::vector<std::list<MAPElite>::const_iterator> uncollapsed(uncollapsed_size, default_iterator);
    std::list<MAPElite>::const_iterator archIter = selection.archive().elites.begin();

    // Populate `uncollapsed` with iterators to the archive by walking the area covered by the selection
    // Cells that are unavailable in the archive remain unchanged in uncollapsed.
    for ( std::list<MAPElite>::const_iterator &element : uncollapsed ) {
        // Update comparator index
        for ( size_t i = 0; i < dimensions; i++ )
            true_index[i] = offsets[i] + offset_index[i];

        // Advance archive iterator
        while ( archIter->bin < true_index )
            ++archIter;

        // Insert archive iterator into uncollapsed
        if ( archIter->bin == true_index )
            element = archIter;

        // Advance index
        for ( int i = dimensions-1; i >= 0; i++ ) {
            if ( ++offset_index[i] % sizes[i] == 0 ) {
                offset_index[i] = 0;
            } else {
                break;
            }
        }
    }

    // If no dimension needs explicit collapsing, our work here is done
    if ( collapsed_size == uncollapsed_size ) {
        selection.selection = std::move(uncollapsed);
        return;
    }

    selection.selection = std::vector<std::list<MAPElite>::const_iterator>(collapsed_size, default_iterator);

    // Reset uncollapsed index
    for ( size_t &o : offset_index )
        o = 0;

    size_t collapsed_index = 0;
    for ( std::list<MAPElite>::const_iterator &element : uncollapsed ) {
        // Take the final selection that we're collapsing to
        std::list<MAPElite>::const_iterator &collapsed = selection.selection.at(collapsed_index);

        // Collapse element onto the final selection if it's better
        if ( element != default_iterator && (collapsed == default_iterator || element->stats.fitness > collapsed->stats.fitness) )
            collapsed = element;

        // Advance uncollapsed index, corresponding to next element
        for ( int i = dimensions-1; i >= 0; i++ ) {
            if ( ++offset_index[i] % sizes[i] == 0 ) {
                offset_index[i] = 0;
            } else {
                break;
            }
        }

        // Update collapsed index
        size_t multiplier = 1;
        collapsed_index = 0;
        for ( size_t i = 0; i < dimensions; i++ ) {
            collapsed_index = offset_index[i] * multiplier;
            multiplier *= selection.width(i);
        }
    }
}



const QString WavegenSelector::action = QString("select");
const quint32 WavegenSelector::magic = 0xa54f3955;
const quint32 WavegenSelector::version = 100;

void WavegenSelector::save(const WavegenSelection &selection)
{
    m_selections.push_back(selection);
    WavegenSelection &sel = m_selections.back();
    finalise(sel);

    QFile file(session.log(this, action));
    QDataStream os;
    if ( !openSaveStream(file, os, magic, version) )
        return;

    // Save archive index in reverse to prevent conflicts during session merge
    os << quint32(session.wavegen().archives().size() - sel.archive_idx);
    os << quint32(sel.ranges.size());
    for ( WavegenSelection::Range const& r : sel.ranges )
        os << quint32(r.min) << quint32(r.max) << r.collapse;
}

void WavegenSelector::load(const QString &action, const QString &, QFile &results)
{
    if ( action != this->action )
        throw std::runtime_error(std::string("Unknown action: ") + action.toStdString());
    QDataStream is;
    quint32 ver = openLoadStream(results, is, magic);
    if ( ver < 100 || ver > version )
        throw std::runtime_error(std::string("File version mismatch: ") + results.fileName().toStdString());

    quint32 idx, nranges, min, max;
    is >> idx >> nranges;
    m_selections.push_back(select(session.wavegen().archives().size() - idx));
    m_selections.back().ranges.resize(nranges);
    for ( WavegenSelection::Range &r : m_selections.back().ranges ) {
        is >> min >> max >> r.collapse;
        r.min = min;
        r.max = max;
    }
    finalise(m_selections.back());
}
