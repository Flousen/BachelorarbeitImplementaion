#ifndef HPC_MKLBLAS_QRBLKREF_HPP
#define HPC_MKLBLAS_QRBLKREF_HPP

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
geqrf(MKL_INT m, MKL_INT n,
       double* a, MKL_INT lda,
       double* tau, double* work,
       MKL_INT lwork, MKL_INT *info )
{
  dgeqrf(&m, &n, a, &lda, tau, work, &lwork, info );
}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_blk_ref(MatrixA &&A, VectorTau &&tau)
{
  assert(tau.inc()==1);
  hpc::matvec::GeMatrix<double> work (A.numRows()*64, A.numCols()*64);
  long long int info = 0;
  geqrf(A.incRow(), A.incCol(), A.data(), A.incCol(),
         tau.data(), work.data(), work.incCol(), &info);

  if (info!=0)
    fmt::printf("Obacht LAPACK info illegal value at: %d\n", info);

}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MM_HPP
