#ifndef HPC_MKLBLAS_SM_HPP
#define HPC_MKLBLAS_SM_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace matvec {


void
trsm(char side, char uplo, char transa, char diag, MKL_INT m, MKL_INT n,
     double alpha, const double *a, MKL_INT lda,
     double *b, const MKL_INT ldb)
{
    dtrsm(&side, &uplo, &transa, &diag, &m, &n, &alpha, a, &lda, b, &ldb);
}

template <typename Alpha,
          typename T, template<typename> class MatrixA,
                      typename                 MatrixB,
          Require< Tr<MatrixA<T>>,
                   Ge<MatrixB>,
                   SameElementType<MatrixA<T>, MatrixB>
                 > = true>
void
sm(const Alpha &alpha, const MatrixA<T> &A, MatrixB &&B)
{
    auto m = assertEqual(A.numRows(), A.numCols(), B.numRows());
    auto n = B.numCols();

    char side   = B.incRow()==1 ? 'L' : 'R';
    char transa = A.incRow()==1 ? 'N' : 'T';
    char uplo   = A.incRow()==1 ? A.is_lower() ? 'L' : 'U'
                                : A.is_lower() ? 'U' : 'L';
    char diag   = A.is_unit()   ? 'U' : 'N';

    assert(std::min(A.incRow(), A.incCol())==1);
    assert(std::min(B.incRow(), B.incCol())==1);

    trsm(side, uplo, transa, diag, m, n, alpha, A.data(), A.incCol(),
         B.data(), B.incCol());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_SM_HPP
