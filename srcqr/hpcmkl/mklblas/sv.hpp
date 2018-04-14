#ifndef HPC_MKLBLAS_SV_HPP
#define HPC_MKLBLAS_SV_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void
trsv(char uplo, char trans, char diag, MKL_INT n, const double *a, MKL_INT lda,
     double *x, MKL_INT incx)
{
    dtrsv(&uplo, &trans, &diag, &n, a, &lda, x, &incx);
}

template <typename T, template<typename> class MatrixA,
                      typename                 VectorB,
          Require< Tr<MatrixA<T>>,
                   Dense<VectorB>,
                   SameElementType<MatrixA<T>, VectorB>
                 > = true>
void
sv(const MatrixA<T> &A, VectorB &&b)
{
    auto n = assertEqual(A.numRows(), A.numCols(), b.length());

    char transa = A.incRow()==1 ? 'N' : 'T';
    char uplo   = A.incRow()==1 ? A.is_lower() ? 'L' : 'U'
                                : A.is_lower() ? 'U' : 'L';
    char diag   = A.is_unit()   ? 'U' : 'N';
    assert(std::min(A.incRow(), A.incCol())==1);

    trsv(uplo, transa, diag, n, A.data(), A.incCol(), b.data(), b.inc());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_SV_HPP
