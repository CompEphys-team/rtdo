#include "clustering.h"
#include "algorithm"


double scalarProduct(const Section &a, const Section &b, int nParams)
{
    double sumA=0, sumB=0, dotp=0;
    for ( int i = 0; i < nParams; i++ ) {
        sumA += a.deviations[i] * a.deviations[i];
        sumB += b.deviations[i] * b.deviations[i];
        dotp += a.deviations[i] * b.deviations[i];
    }
    double denom = sqrt(sumA) * sqrt(sumB);
    return denom==0 ? 0 : dotp/denom;
}

void constructSections(const scalar *diagDelta, int start, int end, int nTraces, std::vector<double> norm, int stride, std::vector<Section> &sections)
{
    const scalar *delta = diagDelta + start*nTraces;
    Section sec {start, 0, std::vector<double>(nTraces-1, 0)};
    for ( int t = start; t <= end; t++ ) {
        ++delta; // Skip current trace

        // Populate values
        for ( int i = 0; i < nTraces-1; i++, delta++ ) {
            double val = *delta / norm[i];
            sec.deviations[i] += val;
        }

        if ( (t-start+1) % stride == 0 ) {
            sec.end = t;
            sections.push_back(sec);
            sec.start = t;
            sec.deviations.assign(nTraces-1, 0);
        }
    }
    if ( sec.start < end ) {
        sec.end = end;
        sections.push_back(sec);
    }
}

std::vector<std::vector<double> > constructSimilarityTable(const std::vector<Section> &sections, int nParams)
{
    std::vector<std::vector<double>> similarity(sections.size());
    for ( size_t i = 0; i < sections.size(); i++ ) {
        similarity[i].resize(i);
        for ( size_t j = 0; j < i; j++ ) {
            similarity[i][j] = scalarProduct(sections[i], sections[j], nParams);
        }
    }
    return similarity;
}

std::vector<Section> extractLargestCluster(std::vector<std::vector<double> > &similarity, std::vector<Section> &sections, double threshold)
{
    // For each section, count the number of above-threshold-similar sections ("siblings")
    std::vector<int> nSiblings(sections.size(), 0);
    for ( size_t i = 0; i < sections.size(); i++ ) {
        for ( size_t j = 0; j < sections.size(); j++ ) {
            if ( j < i ) // Run down the i:th column until reaching the diagonal
                nSiblings[i] += (similarity[i][j] > threshold);
            else if ( j > i ) // Then, run along the i:th row from the diagonal to the right
                nSiblings[i] += (similarity[j][i] > threshold);
        }
    }

    // Take the section with the largest number of siblings, turn the family into a cluster
    auto it = std::max_element(nSiblings.begin(), nSiblings.end());
    size_t rep = it - nSiblings.begin();
    std::vector<Section> cluster;
    std::vector<size_t> clusterIdx;
    cluster.reserve(1 + *it);
    clusterIdx.reserve(1 + *it);
    for ( size_t j = 0; j < sections.size(); j++ ) {
        bool isSibling = true; // the representative (j==rep) goes in by default
        if ( j < rep ) {
            isSibling = (similarity[rep][j] > threshold);
        } else if ( j > rep ) {
            isSibling = (similarity[j][rep] > threshold);
        }
        if ( isSibling ) {
            cluster.push_back(std::move(sections[j]));
            clusterIdx.push_back(j);
        }
    }

    // Remove cluster from table
    for ( auto cit = clusterIdx.rbegin(); cit != clusterIdx.rend(); ++cit ) {
        size_t i = *cit;
        // Delete the i:th row from the diagonal to the right
        for ( size_t j = i+1; j < similarity.size(); j++ )
            similarity[j].erase(similarity[j].begin() + i);

        // Delete the i:th column
        similarity.erase(similarity.begin() + i);

        // Delete the section
        sections.erase(sections.begin() + i);
    }

    return cluster;
}

std::vector<std::pair<int, int>> observeNoSteps(iStimulation iStim, int blankCycles)
{
    int tStart = 0;
    std::vector<std::pair<int,int>> segments;
    for ( const auto step : iStim ) {
        if ( step.t > iStim.duration )
            break;
        if ( !step.ramp ) {
            if ( tStart < step.t )
                segments.push_back(std::make_pair(tStart, step.t));
            tStart = step.t + blankCycles;
        }
    }
    if ( tStart < iStim.duration )
        segments.push_back(std::make_pair(tStart, iStim.duration));
    return segments;
}

std::vector<Section> constructSectionPrimitives(iStimulation iStim, scalar *diagDelta, int blankCycles, int nTraces, std::vector<double> norm, int stride)
{
    std::vector<std::pair<int,int>> segments = observeNoSteps(iStim, blankCycles);
    int nSec = 0;
    for ( std::pair<int,int> const& seg : segments )
        nSec += (seg.second-seg.first)/stride + 1;

    std::vector<Section> sections;
    sections.reserve(nSec);
    for ( std::pair<int,int> &seg : segments ) {
        constructSections(diagDelta, seg.first, seg.second, nTraces, norm, stride, sections);
    }
    return sections;
}

std::vector<Section> findSimilarCluster(std::vector<Section> sections, int nParams, double similarityThreshold, Section master)
{
    // Remove primitives that don't resemble master (but flip & keep opposites)
    for ( auto rit = sections.rbegin(); rit != sections.rend(); ++rit ) {
        double dotp = scalarProduct(master, *rit, nParams);
        if ( -dotp > similarityThreshold )
            for ( int i = 0; i < nParams; i++ )
                rit->deviations[i] *= -1;
        else if ( dotp < similarityThreshold )
            sections.erase(rit.base());
    }

    // compact
    std::vector<Section> compact;
    compact.reserve(sections.size()); // conservative estimate
    Section tmp { sections.front().start, sections.front().start, std::vector<double>(nParams, 0) };
    for ( const Section &sec : sections ) {
        if ( sec.start == tmp.end ) {
            tmp.end = sec.end;
            for ( int i = 0; i < nParams; i++ )
                tmp.deviations[i] += sec.deviations[i];
        } else {
            compact.push_back(std::move(tmp));
            tmp = sec;
        }
    }
    compact.push_back(std::move(tmp));
    compact.shrink_to_fit();
    return compact;
}

std::vector<std::vector<Section> > constructClusters(iStimulation iStim, scalar *diagDelta, int blankCycles, int nTraces, std::vector<double> norm,
                                                     int stride, double similarityThreshold, int minClusterSize)
{
    std::vector<Section> sections = constructSectionPrimitives(iStim, diagDelta, blankCycles, nTraces, norm, stride);
    std::vector<std::vector<double>> similarity = constructSimilarityTable(sections, nTraces-1);

    std::vector<std::vector<Section>> clusters;
    while ( !sections.empty() ) {
        std::vector<Section> cluster = extractLargestCluster(similarity, sections, similarityThreshold);
        if ( int(cluster.size()) < minClusterSize )
            break;

        // compact it
        std::vector<Section> compact;
        compact.reserve(cluster.size()); // conservative estimate
        Section tmp { cluster.front().start, cluster.front().start, std::vector<double>(nTraces-1, 0) };
        for ( const Section &sec : cluster ) {
            if ( sec.start == tmp.end ) {
                tmp.end = sec.end;
                for ( int i = 0; i < nTraces-1; i++ )
                    tmp.deviations[i] += sec.deviations[i];
            } else {
                compact.push_back(std::move(tmp));
                tmp = sec;
            }
        }
        compact.push_back(std::move(tmp));
        compact.shrink_to_fit();
        clusters.push_back(std::move(compact));
    }

    return clusters;
}

void printCluster(std::vector<Section> cluster, int nParams, double dt)
{
    std::vector<double> totals(nParams, 0);
    int len = 0;
    for ( Section &sec : cluster ) {
        std::cout << sec.start*dt << " to " << sec.end*dt << '\n';
        for ( int i = 0; i < nParams; i++ )
            totals[i] += sec.deviations[i];
        len += sec.end - sec.start - 1;
    }
    std::cout << "Total duration: " << len*dt << ", mean deviation: ";
    double norm = 0;
    for ( int i = 0; i < nParams; i++ ) {
        std::cout << '\t' << totals[i]/len;
        if ( norm < fabs(totals[i]) )
            norm = fabs(totals[i]);
    }
    std::cout << "\nNormalised weights:\t";
    for ( int i = 0; i < nParams; i++ ) {
        std::cout << '\t' << fabs(totals[i])/norm;
    }
    std::cout << std::endl;
}

std::vector<std::tuple<int, std::vector<double>, std::vector<Section>>> extractSeparatingClusters(
        const std::vector<std::vector<std::vector<Section>>> &clustersByStim, int nParams)
{
    using T = std::tuple<int, std::vector<double>, std::vector<Section>>;
    std::vector<T> clusters, picks;
    std::vector<Section> bookkeeping;
    for ( int i = 0, nStims = clustersByStim.size(); i < nStims; i++ ) {
        for ( auto it = clustersByStim[i].begin(); it != clustersByStim[i].end(); ++it ) {
            std::vector<double> F(nParams, 0);
            Section tmp {0, 0, std::vector<double>(nParams, 0)};
            for ( const Section &sec : *it ) {
                for ( int j = 0; j < nParams; j++ ) {
                    tmp.deviations[j] += sec.deviations[j];
                }
            }
            double norm = 0;
            for ( int j = 0; j < nParams; j++ )
                if ( norm < fabs(tmp.deviations[j]) )
                    norm = fabs(tmp.deviations[j]);
            for ( int j = 0; j < nParams; j++ )
                F[j] = fabs(tmp.deviations[j])/norm;
            clusters.emplace_back(i, std::move(F), *it);
            bookkeeping.push_back(std::move(tmp));
        }
    }

    for ( int i = 0; i < nParams; i++ ) {
        auto it = clusters.begin();
        std::nth_element(clusters.begin(), it, clusters.end(), [=](const T& lhs, const T& rhs){
            const std::vector<double> &a(std::get<1>(lhs)), &b(std::get<1>(rhs));
            double kA(0), kB(0);
            for ( int j = 0; j < nParams; j++ ) {
                if ( j == i )
                    continue;
                kA += a[j]*a[j];
                kB += b[j]*b[j];
            }
            return a[i]*a[i] / kA > b[i]*b[i] / kB;
        });

        picks.push_back(*it);
    }
    return picks;
}
