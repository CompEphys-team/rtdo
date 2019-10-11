/** RTDO: Closed loop neuron model fitting
 *  Copyright (C) 2019 Felix Kern <kernfel+github@gmail.com>
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc.,
 *  51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
**/


#include <thrust/device_vector.h>
#include <thrust/inner_product.h>
#include "lib_definitions.h"

/**
 * @brief collect_dist_and_err collects the current error along with several measures of parameter-space distance between models.
 * @param nSamples size of the timeseries
 * @param params device pointers to the parameter value arrays, ordered by paramIdx
 * @param targetParam target parameter index for the target-only distance measure
 * @param weight tensor for the weighted parameter space distance measure
 * @param error Output: sum of squared current errors
 * @param distance_unweighted Output: unweighted euclidean parameter-space distance
 * @param distance_target_only Output: Parameter-space distance along the targetParam axis only
 * @param distance_weighted Output: Weighted euclidean parameter-space distance ||(params_reference-params_probe) .* weight||
 * @param get_invariants Shortcut; causes @p distance_target_only and @p distance_unweighted to be multiplied with the respective value in @p error
 */
__global__ void collect_dist_and_err(int nSamples, scalar **params, int targetParam, Parameters weight,
                                     scalar *error, /* RMS current error between tuned probe and reference */
                                     scalar *distance_unweighted, /* euclidean param-space distance between tuned probe and reference */
                                     scalar *distance_target_only, /* distance along the target param axis only */
                                     scalar *distance_weighted, /* euclidean distance, with each axis scaled by @a weight */
                                     bool get_invariants)
{
    unsigned int xThread = blockIdx.x * blockDim.x + threadIdx.x; // probe
    unsigned int yThread = blockIdx.y * blockDim.y + threadIdx.y; // reference
    unsigned int x,y;
    if ( xThread < yThread ) { // transpose subdiagonal half of the top-left quadrant to run on the supradiagonal half of bottom-right quadrant
        // the coordinate transformation is equivalent to squashing the bottom-right supradiagonal triangle to the left border,
        // then flipping it up across the midline.
        x = xThread + NMODELS - yThread; // xnew = x + n-y
        y = NMODELS - yThread - 1;       // ynew = n-y - 1
    } else {
        x = xThread;
        y = yThread;
    }

    scalar err = 0.;
    for ( int i = 0; i < nSamples; ++i ) {
        scalar e = dd_timeseries[x + NMODELS*i] - dd_timeseries[y + NMODELS*i];
        err += e*e;
    }
    err = std::sqrt(err / nSamples);

    if ( x != y ) {
        // Addressing: Squish the diagonal out to prevent extra zeroes
        unsigned int idx = xThread + NMODELS*yThread - yThread - (xThread>yThread);

        scalar dist = 0, noweight_dist = 0;
        for ( int i = 0; i < NPARAMS; i++ ) {
            scalar d = (params[i][x] - params[i][y]);
            noweight_dist += d*d;
            d *= weight[i];
            dist += d*d;
        }
        error[idx] = err;
        distance_weighted[idx] = std::sqrt(dist);
        if ( get_invariants ) {
            distance_target_only[idx] = std::fabs(params[targetParam][x] - params[targetParam][y]);
            distance_unweighted[idx] = std::sqrt(noweight_dist);
        } else {
            distance_target_only[idx] = err * std::fabs(params[targetParam][x] - params[targetParam][y]);
            distance_unweighted[idx] = err * std::sqrt(noweight_dist);
        }
    }
}

__global__ void collect_dist_PC(scalar **params, int targetParam, Parameters weight,
                                scalar *distance_unweighted, /* euclidean param-space distance between tuned probe and reference */
                                scalar *distance_target_only, /* distance along the target param axis only */
                                scalar *distance_weighted, /* euclidean distance, with each axis scaled by @a weight */
                                bool get_invariants)
{
    const unsigned int probeIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int refIdx = probeIdx & ~31;
    scalar dist = 0, noweight_dist = 0;
    for ( int i = 0; i < NPARAMS; i++ ) {
        scalar d = params[i][probeIdx] - params[i][refIdx];
        noweight_dist += d*d;
        d *= weight[i];
        dist += d*d;
    }
    distance_weighted[probeIdx] = std::sqrt(dist);
    if ( get_invariants ) {
        distance_target_only[probeIdx] = std::fabs(params[targetParam][probeIdx] - params[targetParam][refIdx]);
        distance_unweighted[probeIdx] = std::sqrt(noweight_dist);
    }
}

void profile_helper(scalar *err, int nData, int nRelevant,
                    double &rho_weighted, double &rho_unweighted, double &rho_target_only,
                    double &grad_weighted, double &grad_unweighted, double &grad_target_only,
                    bool get_invariants, bool invariant_dists_preprocessed, std::vector<double> &invariants)
{
    thrust::device_ptr<scalar> dist_w = thrust::device_pointer_cast(d_prof_dist_w);
    thrust::device_ptr<scalar> error = thrust::device_pointer_cast(err);
    double sum_sq_dist = thrust::inner_product(dist_w, dist_w + nData, dist_w, scalar(0));  // sum(dist^2)
    double sum_sq_err = thrust::inner_product(error, error + nData, error, scalar(0));      // sum(err^2)
    double sum_dist_err = thrust::inner_product(dist_w, dist_w + nData, error, scalar(0));  // sum(dist*err)
    double sum_dist = thrust::reduce(dist_w, dist_w + nData);                               // sum(dist)
    double sum_err = thrust::reduce(error, error + nData);                                  // sum(err)

    // See 10.1109/ISSPIT.2012.6621260
    double dist_sd = std::sqrt((sum_sq_dist + (sum_dist * sum_dist / nRelevant)) / (nRelevant-1));
    double err_sd = std::sqrt((sum_sq_err + (sum_err * sum_err / nRelevant)) / (nRelevant-1));
    rho_weighted = (sum_dist_err - sum_dist * sum_err / nRelevant) / ((nRelevant-1) * dist_sd * err_sd);
    grad_weighted = sum_err / sum_dist;

    double sum_sq_dist_uw, sum_dist_uw_err, sum_dist_uw, dist_uw_sd;
    thrust::device_ptr<scalar> dist_uw = thrust::device_pointer_cast(d_prof_dist_uw);
    if ( get_invariants ) {
        sum_sq_dist_uw = thrust::inner_product(dist_uw, dist_uw + nData, dist_uw, scalar(0));
        sum_dist_uw_err = thrust::inner_product(dist_uw, dist_uw + nData, error, scalar(0));
        sum_dist_uw = thrust::reduce(dist_uw, dist_uw + nData);
        dist_uw_sd = std::sqrt((sum_sq_dist_uw + (sum_dist_uw * sum_dist_uw / nRelevant)) / (nRelevant-1));

        invariants[0] = sum_dist_uw;
        invariants[1] = dist_uw_sd;
    } else {
        if ( invariant_dists_preprocessed )
            sum_dist_uw_err = thrust::reduce(dist_uw, dist_uw + nData); // Preprocessed with dist_uw[i] = err*dist_uw in kernel
        else
            sum_dist_uw_err = thrust::inner_product(dist_uw, dist_uw + nData, error, scalar(0));
        sum_dist_uw = invariants[0];
        dist_uw_sd = invariants[1];
    }
    rho_unweighted = (sum_dist_uw_err - sum_dist_uw * sum_err / nRelevant) / ((nRelevant-1) * dist_uw_sd * err_sd);
    grad_unweighted = sum_err / sum_dist_uw;

    double sum_sq_dist_to, sum_dist_to_err, sum_dist_to, dist_to_sd;
    thrust::device_ptr<scalar> dist_to = thrust::device_pointer_cast(d_prof_dist_to);
    if ( get_invariants ) {
        sum_sq_dist_to = thrust::inner_product(dist_to, dist_to + nData, dist_to, scalar(0));
        sum_dist_to_err = thrust::inner_product(dist_to, dist_to + nData, error, scalar(0));
        sum_dist_to = thrust::reduce(dist_to, dist_to + nData);
        dist_to_sd = std::sqrt((sum_sq_dist_to + (sum_dist_to * sum_dist_to / nRelevant)) / (nRelevant-1));

        invariants[2] = sum_dist_to;
        invariants[3] = dist_to_sd;
    } else {
        if ( invariant_dists_preprocessed )
            sum_dist_to_err = thrust::reduce(dist_to, dist_to + nData);
        else
            sum_dist_to_err = thrust::inner_product(dist_to, dist_to + nData, error, scalar(0));
        sum_dist_to = invariants[2];
        dist_to_sd = invariants[3];
    }
    rho_target_only = (sum_dist_to_err - sum_dist_to * sum_err / nRelevant) / ((nRelevant-1) * dist_to_sd * err_sd);
    grad_target_only = sum_err / sum_dist_to;
}

extern "C" void profile(int nSamples, const std::vector<AdjustableParam> &params, size_t targetParam, std::vector<scalar> weight,
                        double &rho_weighted, double &rho_unweighted, double &rho_target_only,
                        double &grad_weighted, double &grad_unweighted, double &grad_target_only,
                        std::vector<double> &invariants,
                        bool VC)
{
    scalar *d_params[NPARAMS];
    Parameters weightP;
    for ( size_t i = 0; i < NPARAMS; i++ ) {
        d_params[i] = params[i].d_v;
        weightP[i] = weight[i];
    }
    scalar **dd_params;
    CHECK_CUDA_ERRORS(cudaMalloc((void ***)&dd_params,NPARAMS*sizeof(scalar*)));
    CHECK_CUDA_ERRORS(cudaMemcpy(dd_params,d_params,NPARAMS*sizeof(scalar*),cudaMemcpyHostToDevice));

    bool get_invariants = invariants.empty();
    if ( get_invariants )
        invariants.resize(4);

    if ( VC ) {
        dim3 block(32, 16);
        dim3 grid(NMODELS/32, NMODELS/32);
        collect_dist_and_err<<<grid, block>>>(nSamples, dd_params, targetParam, weightP,
                                              d_prof_error, d_prof_dist_uw, d_prof_dist_to, d_prof_dist_w,
                                              get_invariants);
        profile_helper(d_prof_error, profSz, profSz, rho_weighted, rho_unweighted, rho_target_only,
                       grad_weighted, grad_unweighted, grad_target_only, get_invariants, true, invariants);
    } else {
        collect_dist_PC<<<NMODELS/64, 64>>>(dd_params, targetParam, weightP,
                                            d_prof_dist_uw, d_prof_dist_to, d_prof_dist_w, get_invariants);
        profile_helper(d_summary, NMODELS, 31*NMODELS/32, rho_weighted, rho_unweighted, rho_target_only,
                       grad_weighted, grad_unweighted, grad_target_only, get_invariants, false, invariants);
    }

    CHECK_CUDA_ERRORS(cudaFree((void **)dd_params));
}



__global__ void collect_target_only_dist(scalar *target_params, scalar *dist)
{
    unsigned int xThread = blockIdx.x * blockDim.x + threadIdx.x; // probe
    unsigned int yThread = blockIdx.y * blockDim.y + threadIdx.y; // reference
    unsigned int x,y;
    if ( xThread < yThread ) { // transpose subdiagonal half of the top-left quadrant to run on the supradiagonal half of bottom-right quadrant
        // the coordinate transformation is equivalent to squashing the bottom-right supradiagonal triangle to the left border,
        // then flipping it up across the midline.
        x = xThread + NMODELS - yThread; // xnew = x + n-y
        y = NMODELS - yThread - 1;       // ynew = n-y - 1
    } else {
        x = xThread;
        y = yThread;
    }

    if ( x != y ) {
        unsigned int idx = xThread + NMODELS*yThread - yThread - (xThread>yThread);
        dist[idx] = std::fabs(target_params[x] - target_params[y]);
    }
}

extern "C" double get_mean_distance(const AdjustableParam &p)
{
    dim3 block(32, 16);
    dim3 grid(NMODELS/32, NMODELS/32);
    collect_target_only_dist<<<grid, block>>>(p.d_v, d_prof_dist_to);
    thrust::device_ptr<scalar> dist_to = thrust::device_pointer_cast(d_prof_dist_to);
    double sum_dist_to = thrust::reduce(dist_to, dist_to + profSz);
    return sum_dist_to / profSz;
}
