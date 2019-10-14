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


#include "lib_definitions.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>

static scalar *d_params = nullptr;

void free_cusolver()
{
    cusolverDnDestroy(cusolverH);

    CHECK_CUDA_ERRORS(cudaFree(PCA_d_U));
    CHECK_CUDA_ERRORS(cudaFree(PCA_d_S));
    CHECK_CUDA_ERRORS(cudaFree(PCA_d_VT));
    CHECK_CUDA_ERRORS(cudaFree(PCA_d_lwork));

    CHECK_CUDA_ERRORS(cudaFreeHost(PCA_TL));
    CHECK_CUDA_ERRORS(cudaFree(d_params));
}

void copy_param(int i, scalar *d_v)
{
    if ( !d_params )
        CHECK_CUDA_ERRORS(cudaMalloc((void**)&d_params, NMODELS * NPARAMS * sizeof(scalar)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_params + i*NMODELS, d_v, NMODELS*sizeof(scalar), cudaMemcpyDeviceToDevice))
}

template <typename T>
struct Subtract {
    T arg;
    Subtract(T arg) : arg(arg) {}
    __host__ __device__ T operator() (const T &operand) const { return operand - arg; }
};

template <typename T>
struct Multiply {
    T arg;
    Multiply(T arg) : arg(arg) {}
    __host__ __device__ T operator() (const T &operand) const { return operand * arg; }
};

void singular_value_decomposition(scalar *d_A, int lda, int m, int n, scalar *S)
{
    int lwork = 0;
    int info_gpu = 0;

    // Prepare output space
    resizeArray(PCA_d_S, PCA_S_size, n);
    resizeArray(PCA_d_U, PCA_U_size, m * m);
    resizeArray(PCA_d_VT, PCA_VT_size, lda * n);

    int *devInfo;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&devInfo, sizeof(int)))
    CHECK_CUDA_ERRORS(cudaMemcpy(devInfo, &info_gpu, sizeof(int), cudaMemcpyHostToDevice))

    // Prepare working space
    CUSOLVER_CALL(cusolverDn__scalar__gesvd_bufferSize(cusolverH, m, n, &lwork))
    resizeArray(PCA_d_lwork, PCA_lwork_size, lwork);

    // Compute SVD
    signed char jobu = 'S'; // min(m,n) columns of U
    signed char jobvt = 'N'; // no columns of VT
    CUSOLVER_CALL(cusolverDn__scalar__gesvd(
        cusolverH,
        jobu,
        jobvt,
        m,
        n,
        d_A,
        lda,
        PCA_d_S,
        PCA_d_U,
        m,  // ldu
        PCA_d_VT,
        lda, // ldvt,
        PCA_d_lwork,
        lwork,
        nullptr,
        devInfo));

    // Check status
    CHECK_CUDA_ERRORS(cudaMemcpy(&info_gpu, devInfo, sizeof(int), cudaMemcpyDeviceToHost))
    if ( info_gpu != 0 ) {
        std::cerr << "SVD failed, info = " << info_gpu << std::endl;
    }
    CHECK_CUDA_ERRORS(cudaFree(devInfo));

    // Pull singular values
    CHECK_CUDA_ERRORS(cudaMemcpy(S, PCA_d_S, sizeof(scalar) * n, cudaMemcpyDeviceToHost))
}

/// Performs principal component analysis on @a d_X using SVD, projects the data onto the first @a L components, and returns a vector of (all) singular values.
/// The projection is stored in the host array PCA_TL.
extern "C" std::vector<scalar> principal_components(int L, /* Number of components to project onto */
                                                    scalar *d_X, /* data on device - usually pointer to the first AdjustableParam */
                                                    int ldX = NMODELS, /* Leading dimension of X (total number of rows in X, >= n) */
                                                    int n = NMODELS, /* number of rows = data items */
                                                    int p = NPARAMS) /* number of columns = dimensions */
{
    if ( !d_X )
        d_X = d_params;
    thrust::device_ptr<scalar> X = thrust::device_pointer_cast(d_X);

    // column-wise mean subtraction
    for ( int i = 0; i < p; i++ ) {
        scalar sum = thrust::reduce(X + ldX*i, X + ldX*i + n);
        Subtract<scalar> sub(sum / n);
        thrust::for_each(X + ldX*i, X + ldX*i + n, sub);
    }

    // SVD
    std::vector<scalar> S(p);
    singular_value_decomposition(d_X, ldX, n, p, S.data());

    // Project onto first L components
    // Note, singular_value_decomposition has has PCA_d_U sized as n*n, not ldX*n
    thrust::device_ptr<scalar> U(PCA_d_U);
    for ( int i = 0; i < L; i++ ) {
        Multiply<scalar> mult(S[i]);
        thrust::for_each(U + n*i, U + n*(i+1), mult);
    }

    // Copy to output
    resizeHostArray(PCA_TL, PCA_TL_size, n*L);
    CHECK_CUDA_ERRORS(cudaMemcpy(PCA_TL, PCA_d_U, sizeof(scalar) * n*L, cudaMemcpyDeviceToHost));

    return S;
}
