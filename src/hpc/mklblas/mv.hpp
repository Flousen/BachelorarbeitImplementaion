#ifndef HPC_MKLBLAS_MV_HPP
#define HPC_MKLBLAS_MV_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void
gemv(char trans, MKL_INT m, MKL_INT n, double alpha,
     const double *a, MKL_INT lda, const double *x, MKL_INT incx,
     double beta, double *y, MKL_INT incy)
{
    dgemv(&trans, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy);
}

template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class VectorX,
                      typename                 VectorY,
          Require< Ge<MatrixA<T>>,
                   Dense<VectorX<T>>,
                   Dense<VectorY>,
                   SameElementType<VectorX<T>, VectorY>
                 > = true>
void
mv(const Alpha &alpha, const MatrixA<T> &A, const VectorX<T> &x,
   const Beta &beta, VectorY &&y)
{
    auto m = assertEqual(A.numRows(), y.length());
    auto n = assertEqual(A.numCols(), x.length());

    char transa = A.incRow()==1 ? 'N' : 'T';
    std::ptrdiff_t inc = transa == 'T' ? A.incRow() : A.incCol();

    assert(std::min(A.incRow(), A.incCol())==1);

    gemv(transa, m, n, alpha, A.data(), inc, x.data(), x.inc(),
         beta, y.data(), y.inc());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MV_HPP
