#ifndef HPC_MKLBLAS_TRMM_HPP
#define HPC_MKLBLAS_TRMM_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void trmm(const char side, const char uplo, const char transa,
           const char diag, const MKL_INT m, const MKL_INT n,
           const double alpha, const double *a, const MKL_INT lda,
           double *b, const MKL_INT ldb)
{
    dtrmm(&side, &uplo, &transa, &diag, &m, &n,
          &alpha, a, &lda, b, &ldb);

}

template <typename Alpha,
          typename T, template<typename> class MatrixA,
                      typename MatrixB,
          Require< Tr<MatrixA<T>>,
                   Ge<MatrixB>
                 > = true>
void
mm(const Alpha &alpha, const MatrixA<T> &A, MatrixB &&B, bool transflag = false)
{
    assert(A.numCols() == B.numRows());

    const char side = 'L';
    const char uplo = A.is_lower() ? 'L' : 'U';
    const char transa = A.incRow()==1 ? 'N' : 'T';
    const char diag = A.is_unit() ? 'U' : 'N' ;
    
    trmm(side, uplo, transa, diag, B.numRows(), B.numCols(),
         alpha, A.data(), A.incCol(), B.data(), B.incCol());

}

template <typename Alpha,
          typename T, typename MatrixA,
                      template<typename> class MatrixB,
          Require< Ge<MatrixA>,
                   Tr<MatrixB<T>>
                 > = true>
void
mm(const Alpha &alpha, MatrixA &&A, const MatrixB<T> &B, bool transflag = false)
{
    assert(A.numCols() == B.numRows());
    const char side = 'R';
    const char uplo = B.is_lower() ? 'L' : 'U';
    const char transa = B.incRow()==1 ? 'N' : 'T';
    const char diag = B.is_unit() ? 'U' : 'N' ;
    
    dtrmm(side, uplo, transa, diag, A.numRows(), A.numCols(),
          alpha, B.data(), B.incCol(), A.data(), A.incCol());

}


} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_TRMM_HPP
