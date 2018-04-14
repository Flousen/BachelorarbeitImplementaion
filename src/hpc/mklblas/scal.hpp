#ifndef HPC_MKLBLAS_scal_HPP
#define HPC_MKLBLAS_scal_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void
scal(const MKL_INT n, const double a, double *x, const MKL_INT incx)
{
  dscal(&n, &a, x, &incx);
}

template <typename Alpha, typename VectorX,
          Require< Dense<VectorX> > = true>
void
scal(const Alpha &alpha, VectorX &&x)
{
  scal(x.length(), alpha, x.data(), x.inc());
}


} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MV_HPP
