#include "wavegenselector.h"
#include "session.h"

WavegenSelection::WavegenSelection(Session &session, size_t archive_idx, Result r) :
    Result(r),
    session(session),
    archive_idx(archive_idx),
    ranges(session.wavegenData(archive().resultIndex).mapeDimensions.size())
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

double WavegenSelection::rmin(size_t i) const
{
    size_t multiplier = Wavegen::mape_multiplier(archive().precision);
    return session.wavegenData(archive().resultIndex).mapeDimensions.at(i).bin_inverse(ranges.at(i).min, multiplier);
}

double WavegenSelection::rmax(size_t i) const
{
    size_t multiplier = Wavegen::mape_multiplier(archive().precision);
    return session.wavegenData(archive().resultIndex).mapeDimensions.at(i).bin_inverse(ranges.at(i).max, multiplier);
}

const MAPElite* WavegenSelection::data_relative(std::vector<size_t> idx, bool *ok) const
{
    size_t index = 0, multiplier = 1;
    for ( int i = ranges.size()-1; i >= 0; i-- ) {
        if ( !ranges[i].collapse && idx[i] >= width(i) ) {
            if ( ok )
                *ok = false;
            return nullptr;
        }
        index += idx[i]*multiplier;
        multiplier *= width(i);
    }
    auto ret = selection.at(index);
    if ( ok )
        *ok = (ret != nullptr);
    return ret;
}

const MAPElite* WavegenSelection::data_relative(std::vector<double> idx, bool *ok) const
{
    std::vector<size_t> bin(ranges.size());
    size_t multiplier = session.wavegen().mape_multiplier(archive().precision);
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( ranges[i].collapse )
            continue;
        scalar min = session.wavegenData(archive().resultIndex).mapeDimensions[i].bin_inverse( ranges[i].min, multiplier ); // Behaviour-value offset of the selected area
        scalar shifted = min + idx[i]; // Absolute behavioural value
        bin[i] = session.wavegenData(archive().resultIndex).mapeDimensions[i].bin(shifted, multiplier); // guaranteed valid bin for whole archive, but not for selection
    }
    return data_absolute(bin, ok); // Let data_absolute deal with out of range issues
}

const MAPElite* WavegenSelection::data_absolute(std::vector<size_t> idx, bool *ok) const
{
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( ranges[i].collapse ) {
            // ignore collapsed dimension
            idx[i] = 0;
        } else if ( idx[i] < ranges[i].min || idx[i] > ranges[i].max ) {
            // idx out of selected range, fail
            if ( ok )
                *ok = false;
            return nullptr;
        } else {
            // Offset index to range-relative
            idx[i] -= ranges[i].min;
        }
    }
    return data_relative(idx, ok);
}

const MAPElite* WavegenSelection::data_absolute(std::vector<double> idx, bool *ok) const
{
    std::vector<size_t> bin(ranges.size());
    size_t multiplier = session.wavegen().mape_multiplier(archive().precision);
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        bin[i] = session.wavegenData(archive().resultIndex).mapeDimensions[i].bin(idx[i], multiplier);
    }
    return data_absolute(bin, ok);
}

void WavegenSelection::limit(size_t dimension, double min, double max, bool collapse)
{
    MAPEDimension dim = session.wavegenData(archive().resultIndex).mapeDimensions.at(dimension);
    size_t multiplier = session.wavegen().mape_multiplier(archive().precision);
    limit(dimension, Range{dim.bin(min, multiplier), dim.bin(max, multiplier), collapse});
}

void WavegenSelection::limit(size_t dimension, size_t min, size_t max, bool collapse)
{
    limit(dimension, Range{min, max, collapse});
}

void WavegenSelection::limit(size_t dimension, Range range)
{
    MAPEDimension dim = session.wavegenData(archive().resultIndex).mapeDimensions.at(dimension);
    size_t multiplier = session.wavegen().mape_multiplier(archive().precision);
    size_t rmax = multiplier * dim.resolution - 1;
    if ( range.max < range.min )
        range.max = range.min;
    if ( range.max > rmax )
        range.max = rmax;
    if ( range.min > rmax )
        range.min = rmax;
    ranges.at(dimension) = range;
}

void WavegenSelection::finalise()
{
    const size_t dimensions = ranges.size();
    std::list<MAPElite>::const_iterator default_iterator = archive().elites.end();
    size_t uncollapsed_size = 1, collapsed_size = 1;
    std::vector<size_t> offsets, sizes;
    std::vector<size_t> offset_index(dimensions, 0);
    std::vector<size_t> true_index(dimensions);
    for ( Range const& r : ranges ) {
        size_t s(r.max - r.min + 1);
        uncollapsed_size *= s;
        collapsed_size *= r.collapse ? 1 : s;
        offsets.push_back(r.min);
        sizes.push_back(s);
    }
    std::vector<const MAPElite*> uncollapsed(uncollapsed_size, nullptr);
    std::list<MAPElite>::const_iterator archIter = archive().elites.begin();
    nFinal = 0;

    // Populate `uncollapsed` with iterators to the archive by walking the area covered by the selection
    // Cells that are unavailable in the archive remain unchanged in uncollapsed.
    for ( const MAPElite* &element : uncollapsed ) {
        // Update comparator index
        for ( size_t i = 0; i < dimensions; i++ )
            true_index[i] = offsets[i] + offset_index[i];

        // Advance archive iterator
        while ( archIter != default_iterator && archIter->bin < true_index )
            ++archIter;

        // Stop on exhausted archive
        if ( archIter == default_iterator )
            break;

        // Insert archive iterator into uncollapsed
        if ( archIter->bin == true_index && archIter->fitness >= minFitness ) {
            element = &*archIter;
            ++nFinal;
        }

        // Advance index
        for ( int i = dimensions-1; i >= 0; i-- ) {
            if ( ++offset_index[i] % sizes[i] == 0 ) {
                offset_index[i] = 0;
            } else {
                break;
            }
        }
    }

    // If no dimension needs explicit collapsing, our work here is done
    if ( collapsed_size == uncollapsed_size ) {
        selection = std::move(uncollapsed);
        return;
    }

    selection = std::vector<const MAPElite*>(collapsed_size, nullptr);
    nFinal = 0;

    // Reset uncollapsed index
    for ( size_t &o : offset_index )
        o = 0;

    size_t collapsed_index = 0;
    for ( const MAPElite* element : uncollapsed ) {
        // Take the final selection that we're collapsing to
        const MAPElite* &collapsed = selection.at(collapsed_index);

        // Collapse element onto the final selection if it's better
        if ( element != nullptr && (collapsed == nullptr || element->fitness > collapsed->fitness) ) {
            if ( collapsed == nullptr )
                ++nFinal;
            collapsed = element;
        }

        // Advance uncollapsed index, corresponding to next element
        for ( int i = dimensions-1; i >= 0; i-- ) {
            if ( ++offset_index[i] % sizes[i] == 0 ) {
                offset_index[i] = 0;
            } else {
                break;
            }
        }

        // Update collapsed index
        size_t multiplier = 1;
        collapsed_index = 0;
        for ( int i = dimensions-1; i >= 0; i-- ) {
            if ( !ranges[i].collapse ) {
                collapsed_index += offset_index[i] * multiplier;
                multiplier *= width(i);
            }
        }
    }
}

size_t WavegenSelection::getSizeLimit(size_t n, size_t dimension, bool descending)
{
    if ( n >= nFinal )
        return descending ? ranges[dimension].min : ranges[dimension].max;

    std::vector<const MAPElite*> copy;
    copy.reserve(nFinal);
    for ( const MAPElite *e : selection )
        if ( e )
            copy.push_back(e);
    std::sort(copy.begin(), copy.end(), [&](const MAPElite * const &a, const MAPElite * const &b) {
        if ( !a )   return false;
        if ( !b )   return true;
        return descending ? (b->bin[dimension] < a->bin[dimension]) : (a->bin[dimension] < b->bin[dimension]);
    });

    return copy.at(n-1)->bin[dimension];
}

double WavegenSelection::getFitnessSizeLimit(size_t n)
{
    if ( n >= nFinal )
        return minFitness;

    std::vector<const MAPElite*> copy;
    copy.reserve(nFinal);
    for ( const MAPElite *e : selection )
        if ( e )
            copy.push_back(e);
    std::sort(copy.begin(), copy.end(), [](const MAPElite * const &a, const MAPElite * const &b) {
        if ( !a )   return false;
        if ( !b )   return true;
        return a->fitness > b->fitness;
    });

    return copy.at(n-1)->fitness;
}

QString WavegenSelection::prettyName() const
{
    return QString("%1 waves from archive %2")
            .arg(nFinal)
            .arg(archive_idx);
}
