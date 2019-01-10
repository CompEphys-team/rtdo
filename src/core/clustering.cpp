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

void constructSections(const std::vector<scalar*> pDelta, int dStride, int tStart, int tEnd, std::vector<double> norm, int secLen, std::vector<Section> &sections)
{
    Section sec {tStart, 0, std::vector<double>(pDelta.size(), 0)};
    for ( int t = tStart; t < tEnd; t++ ) {

        // Populate values
        for ( size_t i = 0; i < pDelta.size(); i++ ) {
            double val = *(pDelta[i] + t*dStride) / norm[i];
            sec.deviations[i] += val;
        }

        if ( (t-tStart+1) % secLen == 0 ) {
            sec.end = t;
            sections.push_back(sec);
            sec.start = t;
            sec.deviations.assign(pDelta.size(), 0);
        }
    }
    if ( sec.start < tEnd-1 ) {
        sec.end = tEnd-1;
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
                nSiblings[i] += (fabs(similarity[i][j]) > threshold);
            else if ( j > i ) // Then, run along the i:th row from the diagonal to the right
                nSiblings[i] += (fabs(similarity[j][i]) > threshold);
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
        double sim = 1; // the representative (j==rep) goes in by default
        if ( j < rep ) {
            sim = similarity[rep][j];
        } else if ( j > rep ) {
            sim = similarity[j][rep];
        }
        if ( fabs(sim) > threshold ) {
            if ( sim < 0 )
                for ( double &dev : sections[j].deviations )
                    dev *= -1;
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

iObservations iObserveNoSteps(iStimulation iStim, int blankCycles)
{
    int tStart = 0;
    unsigned int nextObs = 0;
    iObservations obs = {{}, {}};
    for ( const auto step : iStim ) {
        if ( step.t > iStim.duration )
            break;
        if ( !step.ramp ) {
            if ( tStart < step.t ) {
                obs.start[nextObs] = tStart;
                obs.stop[nextObs] = step.t;
                if ( ++nextObs == iObservations::maxObs )
                    break;
            }
            tStart = step.t + blankCycles;
        }
    }
    if ( nextObs < iObservations::maxObs && tStart < iStim.duration ) {
        obs.start[nextObs] = tStart;
        obs.stop[nextObs] = iStim.duration;
    }
    return obs;
}

std::vector<Section> constructSectionPrimitives(iStimulation iStim, std::vector<scalar*> pDelta, int dStride, int blankCycles, std::vector<double> norm, int secLen)
{
    std::vector<std::pair<int,int>> segments = observeNoSteps(iStim, blankCycles);
    int nSec = 0;
    for ( std::pair<int,int> const& seg : segments )
        nSec += (seg.second-seg.first)/secLen + 1;

    std::vector<Section> sections;
    sections.reserve(nSec);
    for ( std::pair<int,int> &seg : segments ) {
        constructSections(pDelta, dStride, seg.first, seg.second, norm, secLen, sections);
    }
    return sections;
}

std::vector<Section> findSimilarCluster(const std::vector<Section> &sections, int nParams, double similarityThreshold, Section master)
{
    // Remove primitives that don't resemble master (but flip & keep opposites)
    std::vector<Section> selection;
    selection.reserve(sections.size());
    for ( const Section &sec : sections ) {
        double dotp = scalarProduct(master, sec, nParams);
        if ( dotp > similarityThreshold || -dotp > similarityThreshold ) {
            selection.push_back(sec);
            if ( std::signbit(dotp) )
                for ( double &d : selection.back().deviations )
                    d *= -1;
        }
    }

    // compact
    std::vector<Section> compact;
    compact.reserve(selection.size()); // conservative estimate
    Section tmp { selection.front().start, selection.front().start, std::vector<double>(nParams, 0) };
    for ( const Section &sec : selection ) {
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

std::vector<std::vector<Section>> constructClusters(iStimulation iStim, std::vector<scalar*> pDelta, int dStride, int blankCycles,
                                                    std::vector<double> norm, int secLen, double similarityThreshold, int minClusterSize)
{
    std::vector<Section> sections = constructSectionPrimitives(iStim, pDelta, dStride, blankCycles, norm, secLen);
    std::vector<std::vector<double>> similarity = constructSimilarityTable(sections, pDelta.size());

    std::vector<std::vector<Section>> clusters;
    while ( !sections.empty() ) {
        std::vector<Section> cluster = extractLargestCluster(similarity, sections, similarityThreshold);
        if ( int(cluster.size()) < minClusterSize )
            break;

        // compact it
        std::vector<Section> compact;
        compact.reserve(cluster.size()); // conservative estimate
        Section tmp { cluster.front().start, cluster.front().start, std::vector<double>(pDelta.size(), 0) };
        for ( const Section &sec : cluster ) {
            if ( sec.start == tmp.end ) {
                tmp.end = sec.end;
                for ( size_t i = 0; i < pDelta.size(); i++ )
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
    std::cout << "Total duration: " << len*dt << ", normalised deviation: ";
    double wNorm = 0, norm = 0;
    for ( int i = 0; i < nParams; i++ )
        norm += totals[i] * totals[i];
    norm = std::sqrt(norm);
    for ( int i = 0; i < nParams; i++ ) {
        std::cout << '\t' << totals[i]/norm;
        if ( wNorm < fabs(totals[i]) )
            wNorm = fabs(totals[i]);
    }
    std::cout << "\nNormalised weights:\t";
    for ( int i = 0; i < nParams; i++ ) {
        std::cout << '\t' << fabs(totals[i])/wNorm;
    }
    std::cout << std::endl;
}

std::vector<std::tuple<int, std::vector<double>, std::vector<Section>>> extractSeparatingClusters(
        const std::vector<std::vector<std::vector<Section>>> &clustersByStim, int nParams)
{
    using T = std::tuple<int, std::vector<double>, std::vector<Section>>;
    std::vector<T> clusters, picks;
    for ( int i = 0, nStims = clustersByStim.size(); i < nStims; i++ ) {
        for ( auto it = clustersByStim[i].begin(); it != clustersByStim[i].end(); ++it ) {
            std::vector<double> F(nParams, 0);
            for ( const Section &sec : *it ) {
                for ( int j = 0; j < nParams; j++ ) {
                    F[j] += sec.deviations[j];
                }
            }

            double sumSquares = 0;
            for ( int j = 0; j < nParams; j++ ) {
                F[j] *= F[j];
                sumSquares += F[j];
            }
            for ( int j = 0; j < nParams; j++ )
                F[j] = std::sqrt(F[j] / sumSquares);

            clusters.emplace_back(i, std::move(F), *it);
        }
    }

    for ( int i = 0; i < nParams; i++ ) {
        auto it = clusters.begin();
        std::nth_element(clusters.begin(), it, clusters.end(), [=](const T& lhs, const T& rhs){
            return std::get<1>(lhs)[i] > std::get<1>(rhs)[i];
        });
        picks.push_back(*it);
    }
    return picks;
}

std::vector<std::vector<scalar*>> getDetunedDiffTraces(const std::vector<Stimulation> &astims, UniversalLibrary &lib, const RunData &rd)
{
    size_t nStims = std::min(astims.size(), maxDetunedDiffTraceStims(lib));
    std::vector<iStimulation> stims(nStims);
    for ( size_t i = 0; i < nStims; i++ )
        stims[i] = iStimulation(astims[i], rd.dt);
    return getDetunedDiffTraces(stims, lib, rd);
}

std::vector<std::vector<scalar*>> getDetunedDiffTraces(const std::vector<iStimulation> &stims, UniversalLibrary &lib, const RunData &rd)
{
    int nParams = lib.adjustableParams.size();
    size_t nStims = std::min(stims.size(), maxDetunedDiffTraceStims(lib));

    lib.setSingularRund();
    lib.simCycles = rd.simCycles;
    lib.integrator = rd.integrator;
    lib.setRundata(0, rd);

    lib.setSingularStim(false);

    size_t warpOffset = 0, tid = 0;
    std::vector<std::vector<scalar*>> pDelta(nStims, std::vector<scalar*>(nParams, nullptr));

    int maxDuration = 0;
    for ( size_t i = 0; i < nStims; i++ ) {
        maxDuration = std::max(maxDuration, stims[i].duration);
    }
    lib.resizeOutput(maxDuration);

    // Set model-level data: tuned/detuned parameters, stim, obs, settleDuration.
    for ( size_t stimIdx = 0; stimIdx < nStims; stimIdx++ ) {
        iObservations obs {{}, {}};
        obs.stop[0] = stims[stimIdx].duration;

        int nWarps = (nParams+30) / 31; // Lane 0 for base model, up to 31 lanes for detuned models
        int pid = 0;
        for ( int warp = 0; warp < nWarps; warp++ ) {
            for ( int lane = 0; lane < 32; lane++ ) { // Fill the warp - if there are more threads than models, the remaining ones are initial
                tid = (warpOffset + warp)*32 + lane;
                lib.stim[tid] = stims[stimIdx];
                lib.obs[tid] = obs;

                for ( int i = 0; i < nParams; i++ ) {
                    AdjustableParam &p = lib.adjustableParams[i];
                    if ( lane > 0 && i == pid ) { // Detune target parameter
                        scalar newp;
                        if ( p.multiplicative ) {
                            newp = p.initial * (1 + p.adjustedSigma);
                            if ( newp > p.max || newp < p.min )
                                newp = p.initial * (1 - p.adjustedSigma);
                        } else {
                            newp = p.initial + p.adjustedSigma;
                            if ( newp > p.max || newp < p.min )
                                newp = p.initial - p.adjustedSigma;
                        }
                        p[tid] = newp;

                        pDelta[stimIdx][pid] = lib.output + tid;
                    } else { // Use initial non-target parameters
                        p[tid] = p.initial;
                    }
                }

                if ( lane > 0 ) // Leave lane 0 fully tuned in all warps
                    ++pid;
            }
        }

        warpOffset += nWarps;
    }

    // Fill the remaining warps/blocks with empty stims
    iStimulation stim = stims[0];
    stim.duration = 0;
    iObservations obs {{}, {}};
    for ( tid = tid+1; tid < lib.NMODELS; ++tid ) {
        lib.stim[tid] = stim;
        lib.obs[tid] = obs;
    }

    lib.assignment = lib.assignment_base | ASSIGNMENT_REPORT_TIMESERIES | ASSIGNMENT_TIMESERIES_COMPARE_LANE0;

    lib.push();
    lib.run();
    lib.pullOutput();
    return pDelta;
}
