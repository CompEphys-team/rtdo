#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "types.h"
#include "universallibrary.h"

// ******************* Helper functions: *****************************

/// Calculates the scalar product between unit-length normalisations of two sections' deviation vector
double scalarProduct(const Section &a, const Section &b, int nParams);

/// Splits the delta record from @a start to @a end into section primitives of length @a secLen (plus a leftovers section at the end),
/// and appends them to @a sections.
void constructSections(const std::vector<scalar*> pDelta, int dStride, int tStart, int tEnd,
                       std::vector<double> norm, int secLen, std::vector<Section> &sections);

/// Constructs an n*n table of scalarProducts between @a sections, represented as triangular matrix.
std::vector<std::vector<double>> constructSimilarityTable(const std::vector<Section> &sections, int nParams);

/// Finds the section with the largest number of siblings with above-threshold similarity, and extracts them in order.
/// Note, @a similarity and @a sections are modified, i.e. the cluster is removed from them.
std::vector<Section> extractLargestCluster(std::vector<std::vector<double>> &similarity, std::vector<Section> &sections, double threshold);

/// Splits the delta record into section primitives of length @a secLen (plus leftovers)
/// while excluding time points within @a blankCycles after a step in @a iStim.
std::vector<Section> constructSectionPrimitives(iStimulation iStim, std::vector<scalar*> pDelta, int dStride,
                                                int blankCycles, std::vector<double> norm, int secLen);

/// Extracts all observation times that exclude hard steps and @a blankCycles thereafter, as (start,end) pairs.
std::vector<std::pair<int, int>> observeNoSteps(iStimulation iStim, int blankCycles);

// ******************** High-level functions: ********************************

/// Finds all sections similar to @a master in @a sections, copying them as a cluster to the @result.
std::vector<Section> findSimilarCluster(std::vector<Section> sections, int nParams, double similarityThreshold, Section master);


/// Extracts all section primitives (via @fn constructSectionPrimitives), clusters them (via @fn extractLargestCluster), discarding clusters with
/// fewer primitives than @a minClusterSize, finally merges adjacent sections and returns the resulting clusters, ordered by size, descending.
std::vector<std::vector<Section>> constructClusters(iStimulation iStim, std::vector<scalar*> pDelta, int dStride, int blankCycles,
                                                    std::vector<double> norm, int secLen, double similarityThreshold, int minClusterSize);


/// Prints a representation of @a cluster to std::cout
void printCluster(std::vector<Section> cluster, int nParams, double dt);


/// Extracts the best-fit single-stimulation cluster for each parameter, where fitness is defined as follows:
/// f = weight_target^2 / sum weight_offtarget^2
/// where weight is the deviation across the cluster, normalised by the largest deviation.
/// @returns a tuple for each parameter, containing, in order:
/// * The stimulation index
/// * The normalised weight of each parameter
/// * The cluster itself.
std::vector<std::tuple<int, std::vector<double>, std::vector<Section>>> extractSeparatingClusters(
        const std::vector<std::vector<std::vector<Section>>> &clustersByStim, int nParams);


/// Runs the given @a stims in @a lib, comparing detuned models to the base model. Returns a collection of pointers to traces for use
/// as pDelta in the functions above; dStride is lib.NMODELS.
std::vector<std::vector<scalar*>> getDetunedDiffTraces(const std::vector<iStimulation> &stims, UniversalLibrary &lib, const RunData &rd);
std::vector<std::vector<scalar*>> getDetunedDiffTraces(const std::vector<Stimulation> &astims, UniversalLibrary &lib, const RunData &rd);

#endif // CLUSTERING_H
