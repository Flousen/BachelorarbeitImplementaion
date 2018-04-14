#ifndef HPC_MKLBLAS_RANK_HPP
#define HPC_MKLBLAS_RANK_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void ger(const MKL_INT m, const MKL_INT n,
          const double alpha,
          const double *x, const MKL_INT incx,
          const double *y, const MKL_INT incy,
          double *a, const MKL_INT lda)
{
  dger(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}


template <typename Alpha,
          typename VectorX,
          typename VectorY,
          typename MatrixA,
          Require< Dense<VectorX>,
                   Dense<VectorY>,
                   Ge<MatrixA>
          > = true>
void
rank1(const Alpha &alpha, const VectorX &x, const VectorY &y,
      MatrixA &&A)
{
  ger(A.numRows(), A.numCols(), alpha,
      x.data(), x.inc(),
      y.data(), y.inc(),
      A.data(), A.incRow());
}

} } // namespace mklblas, hpc
#endif
