#include "profilestats.h"
#include <gsl/gsl_fit.h>
#include <gsl/gsl_statistics_double.h>
#include "session.h"

void computeSingleStats(ProfileStats::Single &s, const ErrorProfile::Profile &profile, const std::vector<double> &values);
void computeFullStats(ProfileStats::Cluster &s);
void computePartialStats(const std::vector<ProfileStats::Single> &vec, double ProfileStats::Single::* member, ProfileStats::Statistic &stat);

ProfileStats::ProfileStats(Session &session) : session(session) {}

void ProfileStats::process(ErrorProfile *parent, int targetParam)
{
    std::vector<std::vector<ErrorProfile::Profile>> profiles = parent->profiles(targetParam);
    stats.resize(profiles.size());

    // Generate a parameter value vector for fitting
    std::vector<double> values(profiles[0][0].size());
    for ( size_t i = 0; i < values.size(); i++ ) {
        values[i] = parent->parameterValue(targetParam, i);
    }

    double minSlope = __DBL_MAX__, maxSlope = 0;
    double minFit = __DBL_MAX__, maxFit = 0;
    double minDeviation = __DBL_MAX__, maxDeviation = 0;
    // `profiles` has one vector<Profile> for each wave; process these separately:
    for ( size_t i = 0; i < stats.size(); i++ ) {
        Cluster &full = stats.at(i);
        const std::vector<ErrorProfile::Profile> &profile = profiles.at(i);
        full.single.resize(profile.size());

        // `profile` contains all profiles for my wave. Let's first get stats for each of these:
        for ( size_t j = 0; j < profile.size(); j++ ) {
            Single &s = full.single.at(j);
            computeSingleStats(s, profile.at(j), values);
            s.deviation = fabs(session.project.model().adjustableParams[targetParam].initial - s.minValue);
        }

        // Then, gather all these single statistics into one mean/median/sd set for full:
        computeFullStats(full);

        // Prepare for ranking
        if ( full.deviation.mean < minDeviation ) minDeviation = full.deviation.mean;
        if ( full.deviation.mean > maxDeviation ) maxDeviation = full.deviation.mean;
        if ( full.slope.mean < minSlope ) minSlope = full.slope.mean;
        if ( full.slope.mean > maxSlope ) maxSlope = full.slope.mean;
        if ( full.slopeFit.mean < minFit ) minFit = full.slopeFit.mean;
        if ( full.slopeFit.mean > maxFit ) maxFit = full.slopeFit.mean;
    }


    // Compute rank index (performance on a statistic normalised by the extant range;
    // this separates well by performance without domain-specific parameters)
    std::vector<Cluster*> rankable;
    rankable.reserve(stats.size());
    for ( Cluster &full : stats ) {
        double deviation_index, slope_index, fit_index;
        deviation_index = (maxDeviation-minDeviation == 0) ? 1 : 1 - (full.deviation.mean - minDeviation) / (maxDeviation-minDeviation);
        slope_index = (maxSlope-minSlope == 0) ? 1 : (full.slope.mean - minSlope) / (maxSlope-minSlope);
        fit_index = (maxFit-minFit == 0 ) ? 1 : 1 - (full.slopeFit.mean - minFit) / (maxFit-minFit);
        full.index = (deviation_index + slope_index + fit_index)/3;
        rankable.push_back(&full);
    }

    // Rank
    std::sort(rankable.begin(), rankable.end(), [](Cluster *lhs, Cluster *rhs){
        if ( lhs->index == rhs->index )
            return lhs->slope.median < rhs->slope.median;
        return lhs->index < rhs->index;
    });
    for ( size_t i = 0; i < rankable.size(); i++ ) {
        rankable[i]->rank = i;
    }
}

void computeSingleStats(ProfileStats::Single &s, const ErrorProfile::Profile &profile, const std::vector<double> &values)
{
    // Find minima and collect errors into a simple vector
    std::vector<double> errors(profile.size());
    size_t minIndex = 0;
    s.minError = profile[minIndex];
    s.localMinima = 0;
    bool downhill = false; // Never count border values as a minimum, except per minValue/minError
    double previousError = 0.;
    size_t i = 0;
    for ( const scalar &error : profile ) {
        errors[i] = error;

        if ( !downhill && error < previousError ) { // Reached a peak
            downhill = true;
        } else if ( downhill && error > previousError ) { // Reached a trough
            ++s.localMinima;
            downhill = false;
            if ( previousError < s.minError ) { // Minimum is (so far) global
                s.minError = previousError;
                minIndex = i-1;
            }
        }
        previousError = error;
        ++i;
    }
    if ( downhill && previousError < s.minError ) { // Right border value is absolute minimum
        minIndex = profile.size()-1;
    }
    s.minValue = values[minIndex];

    // Linear fit on both limbs
    double leftConst, rightConst;
    double leftCov[3], rightCov[3];
    double leftSlope, rightSlope;
    double leftFit, rightFit;
    s.slope = s.slopeFit = 0;
    int limbs = 0;
    if ( minIndex > 0 ) { // Left limb
        gsl_fit_linear(values.data(), 1, errors.data(), 1, minIndex+1,
                       &leftConst, &leftSlope, leftCov, leftCov+1, leftCov+2, &leftFit);
        ++limbs;
        s.slope -= leftSlope; // reverse sign
        s.slopeFit += leftFit;
    }
    if ( minIndex < profile.size()-1 ) { // Right limb
        gsl_fit_linear(values.data()+minIndex, 1, errors.data()+minIndex, 1, profile.size()-minIndex,
                       &rightConst, &rightSlope, rightCov, rightCov+1, rightCov+2, &rightFit);
        ++limbs;
        s.slope += rightSlope;
        s.slopeFit += rightFit;
    }
    s.slope /= limbs;
    s.slopeFit /= limbs;
}

void computeFullStats(ProfileStats::Cluster &s)
{
    computePartialStats(s.single, &ProfileStats::Single::minValue, s.minValue);
    computePartialStats(s.single, &ProfileStats::Single::minError, s.minError);
    computePartialStats(s.single, &ProfileStats::Single::deviation, s.deviation);
    computePartialStats(s.single, &ProfileStats::Single::localMinima, s.localMinima);
    computePartialStats(s.single, &ProfileStats::Single::slope, s.slope);
    computePartialStats(s.single, &ProfileStats::Single::slopeFit, s.slopeFit);
}

void computePartialStats(const std::vector<ProfileStats::Single> &vec, double ProfileStats::Single::* member, ProfileStats::Statistic &stat)
{
    // Of course GSL doesn't include an unsorted median algo, because why would it?
    // So sorting (or implementing some more involved median algo) is required.
    std::vector<double> values(vec.size());
    double sum = 0;
    for ( size_t i = 0; i < values.size(); i++ ) {
        values[i] = vec[i].*member;
        sum += vec[i].*member;
    }
    std::sort(values.begin(), values.end());
    stat.mean = sum/values.size();
    stat.sd = gsl_stats_sd_m(values.data(), 1, values.size(), stat.mean);
    stat.median = gsl_stats_median_from_sorted_data(values.data(), 1, values.size());
}

QDataStream &operator<<(QDataStream &os, const ProfileStats::Statistic &stat)
{
    os << stat.mean << stat.median << stat.sd;
    return os;
}

QDataStream &operator>>(QDataStream &is, ProfileStats::Statistic &stat)
{
    is >> stat.mean >> stat.median >> stat.sd;
    return is;
}

QDataStream &operator>>(QDataStream &is, ProfileStats &s)
{
    qint32 rank;
    quint32 nCluster, nSingle;
    is >> nCluster;
    s.stats.resize(nCluster);
    for ( ProfileStats::Cluster &cl : s.stats ) {
        is >> cl.minValue >> cl.minError >> cl.deviation >> cl.localMinima >> cl.slope >> cl.slopeFit;
        is >> cl.index >> rank >> nSingle;
        cl.rank = rank;
        cl.single.resize(nSingle);
        for ( ProfileStats::Single &si : cl.single )
            is >> si.minValue >> si.minError >> si.deviation >> si.localMinima >> si.slope >> si.slopeFit;
    }
    return is;
}

QDataStream &operator<<(QDataStream &os, const ProfileStats &s)
{
    os << quint32(s.stats.size());
    for ( const ProfileStats::Cluster &cl : s.stats ) {
        os << cl.minValue << cl.minError << cl.deviation << cl.localMinima << cl.slope << cl.slopeFit;
        os << cl.index << qint32(cl.rank);
        os << quint32(cl.single.size());
        for ( const ProfileStats::Single &si : cl.single )
            os << si.minValue << si.minError << si.deviation << si.localMinima << si.slope << si.slopeFit;
    }
    return os;
}
