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
  lwork = -1;
  double temp=0;
  dgeqrf(&m, &n, a, &lda, tau,(double *) &temp, &lwork, info );
  fmt::printf("lwork %d\n",lwork);
  fmt::printf("tempk %d\n",(long long int)temp);
  matvec::DenseVector<double> W((long long int) temp);
  long long int temp2 = temp;
  dgeqrf(&m, &n, a, &lda, tau, W.data(), &temp2, info );

}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_blk_ref(MatrixA &&A, VectorTau &&tau)
{
  assert(tau.inc()==1);
  hpc::matvec::GeMatrix<double> work (A.numRows(), A.numCols());
  long long int info = 0;
  fmt::printf("passiert hier was? %d\n",A.data() );
  geqrf(A.numRows(), A.numCols(), A.data(), A.incCol(),
         tau.data(), work.data(), work.incCol(), &info);

  if (info!=0)
    fmt::printf("LAPACK info illegal value at: %d\n", info);

}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MM_HPP
