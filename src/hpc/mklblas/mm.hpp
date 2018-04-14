#ifndef HPC_MKLBLAS_MM_HPP
#define HPC_MKLBLAS_MM_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace matvec {

void
gemm(char transa, char transb, MKL_INT m, MKL_INT n, MKL_INT k,
     double alpha, const double *a, MKL_INT lda,
     const double *b, MKL_INT ldb,
     double beta, double *c, MKL_INT ldc)
{
    dgemm(&transa, &transb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta,
          c, &ldc);
}


/// Functions `hpc::ulmblas::mm`
/// ---------------------------
///
/// - gemm operation $C \leftarrow \beta C + \alpha A B$
///
template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
                      typename MatrixC,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixB<T>>,
                   Ge<MatrixC>
                 > = true>
void
mm(const Alpha &alpha, const MatrixA<T> &A, const MatrixB<T> &B,
   const Beta &beta, MatrixC &&C)
{
    auto m = assertEqual(C.numRows(), A.numRows());
    auto n = assertEqual(C.numCols(), B.numCols());
    auto k = assertEqual(A.numCols(), B.numRows());

    char transA = A.incRow()==1 ? 'N' : 'T';
    char transB = B.incRow()==1 ? 'N' : 'T';

    assert(std::min(A.incRow(), A.incCol())==1);
    assert(std::min(B.incRow(), B.incCol())==1);
    assert(C.incRow()==1);

    gemm(transA, transB, m, n, k, alpha,
         A.data(), A.incCol(),
         B.data(), B.incCol(),
         beta, C.data(), C.incCol());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MM_HPP
