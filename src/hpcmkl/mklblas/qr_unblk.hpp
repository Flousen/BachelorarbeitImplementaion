#ifndef HPC_MKLBLAS_MM_HPP
#define HPC_MKLBLAS_MM_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void
qr_unblk (int matrix_layout, lapack_int m, lapack_int n,
     double* a, lapack_int lda, double* tau)
{
    //dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,
    //      c, &ldc);
    //LAPACKE_dgeqr2 (int matrix_layout, lapack_int m, lapack_int n,
    //    double* a, lapack_int lda, double* tau);
    LAPACKE_dgeqr2 (matrix_layout, m, n,
                    a, lda, tau);
}


/// Functions `hpc::ulmblas::mm`
/// ---------------------------
///
/// - gemm operation $C \leftarrow \beta C + \alpha A B$
///
template < typename T, typename MatrixA,
                      typename VectorTau>
void
qr_unblk(MatrixA &&A, VectroTau &&tau)
{
    //auto m = assertEqual(C.numRows(), A.numRows());
    //auto n = assertEqual(C.numCols(), B.numCols());

    qr_unblk(LAPACK_COL_MAJOR, A.numRows(), A.numCols(),
        A.data(), A.incRow());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MM_HPP
