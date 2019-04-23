#include "lib_definitions.h"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/for_each.h>

void free_cusolver()
{
    cusolverDnDestroy(cusolverH);

    CHECK_CUDA_ERRORS(cudaFree(PCA_d_U));
    CHECK_CUDA_ERRORS(cudaFree(PCA_d_S));
    CHECK_CUDA_ERRORS(cudaFree(PCA_d_VT));
    CHECK_CUDA_ERRORS(cudaFree(PCA_d_lwork));

    CHECK_CUDA_ERRORS(cudaFreeHost(PCA_TL));
}

template <typename T>
struct Subtract {
    T arg;
    Subtract(T arg) : arg(arg) {}
    __host__ __device__ void operator() (T &operand) { operand -= arg; }
};

template <typename T>
struct Multiply {
    T arg;
    Multiply(T arg) : arg(arg) {}
    __host__ __device__ void operator() (T &operand) { operand *= arg; }
};

void singular_value_decomposition(scalar *d_A, int m, int n, scalar *S)
{
    const int lda = m;

    int lwork = 0;
    int info_gpu = 0;

    // Prepare output space
    resizeArray(PCA_d_S, PCA_S_size, n);
    resizeArray(PCA_d_U, PCA_U_size, lda * m);
    resizeArray(PCA_d_VT, PCA_VT_size, lda * n);

    int *devInfo;
    CHECK_CUDA_ERRORS(cudaMalloc((void**)&devInfo, sizeof(int)))

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
        lda,  // ldu
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
        exit(EXIT_FAILURE);
    }
    CHECK_CUDA_ERRORS(cudaFree(devInfo));

    // Pull singular values
    CHECK_CUDA_ERRORS(cudaMemcpy(S, PCA_d_S, sizeof(scalar) * n, cudaMemcpyDeviceToHost))
}

/// Performs principal component analysis on @a d_X using SVD, projects the data onto the first @a L components, and returns a vector of (all) singular values.
/// The projection is stored in the host array PCA_TL.
extern "C" std::vector<scalar> principal_components(int L, /* Number of components to project onto */
                                                    scalar *d_X, /* data on device - usually pointer to the first AdjustableParam */
                                                    int n = NMODELS, /* number of rows = data items */
                                                    int p = NPARAMS) /* number of columns = dimensions */
{
    thrust::device_ptr<scalar> X = thrust::device_pointer_cast(d_X);

    // column-wise mean subtraction
    for ( int i = 0; i < p; i++ ) {
        scalar sum = thrust::reduce(X + n*i, X + n*(i+1));
        Subtract<scalar> sub(sum / n);
        thrust::for_each(X + n*i, X + n*(i+1), sub);
    }

    // SVD
    std::vector<scalar> S(p);
    singular_value_decomposition(d_X, n, p, S.data());

    // Project onto first L components
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
