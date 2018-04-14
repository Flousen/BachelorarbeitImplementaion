#ifndef HPC_MKLBLAS_dot_HPP
#define HPC_MKLBLAS_dot_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

double
dot(MKL_INT n, const double *x, MKL_INT incx,
     const double *y, MKL_INT incy)
{
  return ddot(&n, x, &incx, y, &incy);
}


template <typename T, template<typename> class VectorX,
                      template<typename> class VectorY,
          Require< Dense<VectorX<T>>,
                   Dense<VectorY<T>> > = true>
T
dot(const VectorX<T> &x, const VectorY<T> &y)
{
    return dot(x.length(), x.data(), x.inc(), y.data(), y.inc());
}


} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MV_HPP
