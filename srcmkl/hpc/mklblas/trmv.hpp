#ifndef HPC_MKLBLAS_TRMV_HPP
#define HPC_MKLBLAS_TRMV_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void
trmv (const char uplo, const char transa, const char diag, const MKL_INT n,
      const double *a, const MKL_INT lda, double *b, const MKL_INT incx)
{
    dtrmv(&uplo, &transa, &diag, &n, a, &lda, b, &incx);
}


template <typename Alpha,
          typename T, template<typename> class MatrixA,
                      template<typename> class VectorX,
          Require< Tr<MatrixA<T>>,
                   Dense<VectorX<T>>
                 > = true>
void
mv(const Alpha &alpha, const MatrixA<T> &A, const VectorX<T> &x)
{
    assert(A.numCols()==x.length());
   
    const char uplo = A.is_lower() ? 'U' : 'L';
    const char trans = A.incRow()==1 ? 'N' : 'T';
    const char diag = A.is_unit() ? 'U' : 'N' ;

    trmv(uplo, trans, diag, A.numCols(),
         A.data(), A.incCol(), x.data(), x.inc());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_TRMV_HPP
