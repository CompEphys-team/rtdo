#include "wavegenselector.h"
#include "session.h"

WavegenSelection::WavegenSelection(Session &session, size_t archive_idx, Result r) :
    Result(r),
    session(session),
    archive_idx(archive_idx),
    ranges(session.wavegenData(archive().resultIndex).mapeDimensions.size()),
    paretoMaximise(ranges.size(), true),
    paretoTolerance(ranges.size(), 1)
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
    const MAPEDimension &dim = archive().searchd(session).mapeDimensions.at(i);
    return dim.bin_inverse(ranges.at(i).min, dim.multiplier(archive().precision));
}

double WavegenSelection::rmax(size_t i) const
{
    const MAPEDimension &dim = archive().searchd(session).mapeDimensions.at(i);
    return dim.bin_inverse(ranges.at(i).max, dim.multiplier(archive().precision));
}

size_t WavegenSelection::index_relative(std::vector<size_t> idx, bool *ok) const
{
    size_t index = 0, multiplier = 1;
    for ( int i = ranges.size()-1; i >= 0; i-- ) {
        if ( !ranges[i].collapse ) {
            if ( ok && idx[i] >= width(i) ) {
                *ok = false;
                return 0;
            }
            index += idx[i]*multiplier;
            multiplier *= width(i);
        }
    }
    return index;
}

const MAPElite* WavegenSelection::data_relative(std::vector<size_t> idx, bool *ok) const
{
    bool idx_ok = true;
    size_t index = index_relative(idx, &idx_ok);
    if ( !idx_ok ) {
        if ( ok ) *ok = false;
        return nullptr;
    }
    auto ret = selection.at(index);
    if ( ok )
        *ok = (ret != nullptr);
    return ret;
}

const MAPElite* WavegenSelection::data_relative(std::vector<double> idx, bool *ok) const
{
    std::vector<size_t> bin(ranges.size());
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( ranges[i].collapse )
            continue;
        const MAPEDimension &dim = archive().searchd(session).mapeDimensions[i];
        scalar min = dim.bin_inverse(ranges[i].min, dim.multiplier(archive().precision)); // Behaviour-value offset of the selected area
        scalar shifted = min + idx[i]; // Absolute behavioural value
        bin[i] = dim.bin(shifted, dim.multiplier(archive().precision)); // guaranteed valid bin for whole archive, but not for selection
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
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        const MAPEDimension &dim = archive().searchd(session).mapeDimensions[i];
        bin[i] = dim.bin(idx[i], dim.multiplier(archive().precision));
    }
    return data_absolute(bin, ok);
}

void WavegenSelection::limit(size_t dimension, double min, double max, bool collapse)
{
    MAPEDimension dim = archive().searchd(session).mapeDimensions.at(dimension);
    size_t multiplier = dim.multiplier(archive().precision);
    limit(dimension, Range{dim.bin(min, multiplier), dim.bin(max, multiplier), collapse});
}

void WavegenSelection::limit(size_t dimension, size_t min, size_t max, bool collapse)
{
    limit(dimension, Range{min, max, collapse});
}

void WavegenSelection::limit(size_t dimension, Range range)
{
    MAPEDimension dim = archive().searchd(session).mapeDimensions.at(dimension);
    size_t rmax = dim.multiplier(archive().precision) * dim.resolution - 1;
    if ( range.max < range.min )
        range.max = range.min;
    if ( range.max > rmax )
        range.max = rmax;
    if ( range.min > rmax )
        range.min = rmax;
    ranges.at(dimension) = range;
}

void WavegenSelection::select_uncollapsed()
{
    const size_t dimensions = ranges.size();
    std::list<MAPElite>::const_iterator default_iterator = archive().elites.end();
    size_t uncollapsed_size = 1;
    std::vector<size_t> offsets, sizes;
    std::vector<size_t> offset_index(dimensions, 0);
    std::vector<size_t> true_index(dimensions);
    for ( Range const& r : ranges ) {
        size_t s(r.max - r.min + 1);
        uncollapsed_size *= s;
        offsets.push_back(r.min);
        sizes.push_back(s);
    }
    selection.assign(uncollapsed_size, nullptr);
    std::list<MAPElite>::const_iterator archIter = archive().elites.begin();
    nFinal = 0;

    // Populate `uncollapsed` with iterators to the archive by walking the area covered by the selection
    // Cells that are unavailable in the archive remain unchanged in uncollapsed.
    for ( const MAPElite* &element : selection ) {
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
}

void WavegenSelection::collapse()
{
    const size_t dimensions = ranges.size();
    size_t collapsed_size = 1;
    std::vector<size_t> sizes;
    std::vector<size_t> offset_index(dimensions, 0);
    for ( Range const& r : ranges ) {
        size_t s(r.max - r.min + 1);
        collapsed_size *= r.collapse ? 1 : s;
        sizes.push_back(s);
    }

    std::vector<const MAPElite*> uncollapsed;
    using std::swap;
    swap(selection, uncollapsed);
    selection.assign(collapsed_size, nullptr);
    nFinal = 0;

    for ( const MAPElite* element : uncollapsed ) {
        // Take the final selection that we're collapsing to
        const MAPElite* &collapsed = selection.at(index_relative(offset_index));

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
    }
}

struct Anchor {
    std::vector<size_t> idx;
    scalar fitness;
    bool dominates(const Anchor &rhs, std::vector<int> direction /* positive: larger preferred */) const
    {
        bool dom = false;
        if ( fitness < rhs.fitness )
            return false;
        else if ( fitness > rhs.fitness )
            dom |= true;
        for ( size_t i = 0; i < idx.size(); i++ ) {
            if ( direction[i] > 0 ) {
                if ( idx[i] < rhs.idx[i] )
                    return false;
                else if ( idx[i] > rhs.idx[i] )
                    dom |= true;
            } else if ( direction[i] < 0 ) {
                if ( idx[i] > rhs.idx[i] )
                    return false;
                else if ( idx[i] < rhs.idx[i] )
                    dom |= true;
            }
        }
        return dom;
    }
};

void WavegenSelection::select_pareto()
{
    const size_t dimensions = ranges.size();
    std::vector<size_t> start(dimensions), end(dimensions);
    std::vector<int> step(dimensions), direction(dimensions);
    std::vector<size_t> probeDims;
    for ( size_t i = 0; i < dimensions; i++ ) {
        const Range &r = ranges[i];
        if ( r.collapse || r.max == r.min ) {
            direction[i] = 0;
            start[i] = 0;
        } else if ( paretoMaximise[i] ) {
            direction[i] = 1;
            step[i] = -1;
            start[i] = width(i);
            end[i] = size_t(0) - 1;
            probeDims.push_back(i);
        } else {
            direction[i] = -1;
            start[i] = 0;
            end[i] = width(i) + 1;
            step[i] = 1;
            probeDims.push_back(i);
        }
    }
    std::vector<Anchor> anchors;
    Anchor probe = {start, 0};
    probe.idx[probeDims.front()] -= step[probeDims.front()]; // Initial position just before start
    scalar latestFitness = 0;
    while ( probe.idx != end ) {
        bool found = false, done = false;
        while ( !found && !done ) {
            for ( size_t i : probeDims ) {
                probe.idx[i] += step[i];
                if ( probe.idx[i] == end[i] ) {
                    if ( i == probeDims.back() ) {
                        done = true;
                        break;
                    }
                    probe.idx[i] = start[i];
                    latestFitness = 0;
                } else {
                    const MAPElite *el = data_relative(probe.idx, &found);
                    if ( found && el->fitness > latestFitness )
                        probe.fitness = el->fitness;
                    else
                        found = false;
                    break;
                }
            }
        }
        if ( done )
            break;

        bool dom = false;
        for ( const Anchor &anchor : anchors ) {
            if ( anchor.dominates(probe, direction) ) {
                dom = true;
                break;
            }
        }
        if ( !dom )
            anchors.push_back(probe);
    }

    // Include the anchors
    std::vector<const MAPElite*> redux(selection.size(), nullptr);
    nFinal = anchors.size();
    for ( const Anchor &anchor : anchors ) {
        size_t index = index_relative(anchor.idx);
        redux[index] = selection[index];
    }

    // Precalculate offsets
    std::vector<int> offset_indices;
    size_t start_index = index_relative(start);
    std::vector<size_t> offset(dimensions, 0);
    size_t nOffsetIndices = 1, n = 0;
    for ( size_t i : probeDims )
        nOffsetIndices *= paretoTolerance[i];
    while ( ++n < nOffsetIndices ) {
        for ( size_t i : probeDims ) {
            if ( ++offset[i] == paretoTolerance[i] ) {
                offset[i] = 0;
            } else {
                break;
            }
        }
        for ( size_t i : probeDims )
            probe.idx[i] = start[i] + step[i]*offset[i];
        offset_indices.push_back(start_index - index_relative(probe.idx));
    }

    // Include non-pareto-optimal points that are near enough to anchors, as defined by depth
    if ( nOffsetIndices > 1 ) {
        for ( const Anchor &anchor : anchors ) {
            size_t nBoundedIndices = 1;
            std::vector<size_t> trueDepth(probeDims.size());
            for ( size_t i : probeDims ) {
                size_t availableDepth = step[i] * ((int)end[i] - (int)anchor.idx[i]);
                trueDepth[i] = std::min(availableDepth, paretoTolerance[i]);
                nBoundedIndices *= trueDepth[i];
            }

            scalar shiftedFitness = anchor.fitness - paretoFitnessTol;
            if ( nBoundedIndices < nOffsetIndices ) {
                offset.assign(dimensions, 0);
                n = 0;
                while ( ++n < nBoundedIndices ) {
                    for ( size_t i : probeDims ) {
                        if ( ++offset[i] == trueDepth[i] ) {
                            offset[i] = 0;
                        } else {
                            break;
                        }
                    }
                    for ( size_t i : probeDims )
                        probe.idx[i] = anchor.idx[i] + step[i]*offset[i];
                    size_t index = index_relative(probe.idx);
                    if ( !redux[index] && selection[index] && selection[index]->fitness >= shiftedFitness ) {
                        redux[index] = selection[index];
                        ++nFinal;
                    }
                }
            } else {
                size_t index = index_relative(anchor.idx);
                for ( int off : offset_indices ) {
                    if ( !redux[index-off] && selection[index-off] && selection[index-off]->fitness >= shiftedFitness ) {
                        redux[index - off] = selection[index-off];
                        ++nFinal;
                    }
                }
            }
        }
    }

    using std::swap;
    swap(redux, selection);
}

void WavegenSelection::finalise()
{
    select_uncollapsed();

    bool needs_collapse = false;
    for ( const Range &r : ranges )
        needs_collapse |= (r.collapse && (r.max > r.min));
    if ( needs_collapse )
        collapse();

    if ( paretoFront )
        select_pareto();
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
