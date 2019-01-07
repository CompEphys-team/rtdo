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

size_t WavegenSelection::index_relative(const std::vector<size_t> &idx, bool *ok) const
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

size_t WavegenSelection::index_absolute(std::vector<size_t> idx, bool *ok) const
{
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( ranges[i].collapse ) {
            // ignore collapsed dimension
            idx[i] = 0;
        } else if ( idx[i] < ranges[i].min || idx[i] > ranges[i].max ) {
            // idx out of selected range, fail
            if ( ok )
                *ok = false;
            return 0;
        } else {
            // Offset index to range-relative
            idx[i] -= ranges[i].min;
        }
    }
    return index_relative(idx);
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
    bool idx_ok = true;
    size_t index = index_absolute(idx, &idx_ok);
    if ( !idx_ok ) {
        if ( ok ) *ok = false;
        return nullptr;
    }
    auto ret = selection.at(index);
    if ( ok )
        *ok = (ret != nullptr);
    return ret;
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

int WavegenSelection::dominatesIntolerant(const MAPElite *lhs, const MAPElite *rhs) const
{
    int dir = 0;
    if ( lhs->fitness > rhs->fitness )
        dir = 1;
    else if ( lhs->fitness < rhs->fitness )
        dir = -1;
    int dom = dir;
    int strictDir = dir;
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( lhs->bin[i] > rhs->bin[i] )
            dir = paretoMaximise[i] ? 1 : -1;
        else if ( lhs->bin[i] < rhs->bin[i] )
            dir = paretoMaximise[i] ? -1 : 1;
        else
            continue;
        if ( dir ) {
            if ( strictDir && dir != strictDir )
                return 0;
            strictDir = dir;
            dom += dir;
        }
    }
    return dom;
}

int WavegenSelection::dominatesTolerant(const MAPElite *lhs, const MAPElite *rhs) const
{
    int dir = 0;
    bool tolerate = true;
    if ( lhs->fitness > rhs->fitness ) {
        dir = 1;
        tolerate &= lhs->fitness <= rhs->fitness + paretoFitnessTol;
    } else if ( lhs->fitness < rhs->fitness ) {
        dir = -1;
        tolerate &= lhs->fitness + paretoFitnessTol >= rhs->fitness;
    }
    int dom = dir;
    int strictDir = dir;
    for ( size_t i = 0; i < ranges.size(); i++ ) {
        if ( lhs->bin[i] > rhs->bin[i] ) {
            dir = paretoMaximise[i] ? 1 : -1;
            tolerate &= lhs->bin[i] <= rhs->bin[i] + paretoTolerance[i]*(paretoMaximise[i] ? 1 : -1);
        } else if ( lhs->bin[i] < rhs->bin[i] ) {
            dir = paretoMaximise[i] ? -1 : 1;
            tolerate &= lhs->bin[i] + paretoTolerance[i]*(paretoMaximise[i] ? 1 : -1) >= rhs->bin[i];
        } else {
            continue;
        }
        if ( dir ) {
            if ( strictDir && dir != strictDir )
                return 0;
            strictDir = dir;
            dom += dir;
        }
    }
    return tolerate ? dir : 2*dom;
}

void WavegenSelection::select_pareto()
{
    bool tolerant = paretoFitnessTol > 0;
    for ( size_t tol : paretoTolerance )
        tolerant &= tol>0;
    auto dominates = tolerant ? &WavegenSelection::dominatesTolerant : &WavegenSelection::dominatesIntolerant;

    std::vector<const MAPElite*> front;
    std::vector<std::vector<const MAPElite*>> flocks;
    for ( const MAPElite *el : selection ) {
        if ( !el )
            continue;
        bool dominated = false;
        std::vector<const MAPElite*> myFlock;
        auto flockIt = flocks.begin();
        for ( auto it = front.begin(); it != front.end(); ++it, ++flockIt ) {
            int dom = (this->*dominates)(el, *it);
            if ( dom > 0 ) { // el dominates it => remove it from front
                if ( tolerant ) {
                    for ( const MAPElite *sheep : *flockIt ) // Add any tolerable sheep to flock
                        if ( (this->*dominates)(el, sheep) == 1 )
                            myFlock.push_back(sheep);
                    if ( dom == 1 )
                        myFlock.push_back(*it);
                    flockIt = flocks.erase(flockIt) - 1;
                }
                it = front.erase(it) - 1;
            } else if ( dom < 0 ) {
                if ( tolerant && dom == -1 )
                    flockIt->push_back(el);
                dominated = true; // it dominates el => cancel search, don't add el
                break;
            }
        }
        if ( !dominated ) {
            front.push_back(el);
            if ( tolerant )
                flocks.push_back(myFlock);
        }
    }

    selection.assign(selection.size(), nullptr);
    nFinal = front.size();
    for ( const MAPElite *f : front )
        selection[index_absolute(f->bin)] = f;
    if ( tolerant ) {
        for ( const std::vector<const MAPElite*> &flock : flocks ) {
            for ( const MAPElite *f : flock )
                selection[index_absolute(f->bin)] = f;
            nFinal += flock.size();
        }
    }
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
