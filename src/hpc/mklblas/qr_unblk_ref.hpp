#ifndef HPC_MKLBLAS_QRUNBREF_HPP
#define HPC_MKLBLAS_QRUNBREF_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl.h>
#include <mkl_types.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>

namespace hpc { namespace mklblas {

void
qr_unblk_blas(MKL_INT m, MKL_INT n, double* a, MKL_INT lda,
         double* tau, double* work, MKL_INT *info )
{
    dgeqr2(&m, &n, a, &lda, tau, work, info);
}

template < typename MatrixA, typename VectorTau,
           Require<Ge<MatrixA>, Dense<VectorTau>> = true>
void
qr_unblk_ref(MatrixA &&A, VectorTau &&tau)
{
    //auto m = assertEqual(C.numRows(), A.numRows());
    //auto n = assertEqual(C.numCols(), B.numCols());
    assert(tau.inc() == 1);
    hpc::matvec::GeMatrix<double> W (A.numRows(), A.numCols());
    long long int info  = 0;

    qr_unblk_blas(A.numRows(), A.numCols(),
        A.data(), A.incCol(), tau.data(), W.data(), &info);
    if (info!=0)
      fmt::printf("Obacht LAPACK info illegal value at: %d\n", info);
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_QR_HPP

