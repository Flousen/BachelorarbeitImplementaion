#ifndef HPC_MKLBLAS_COPY_HPP
#define HPC_MKLBLAS_COPY_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>

namespace hpc { namespace mklblas {

void
copy(MKL_INT n, const double *x, MKL_INT incx,
      double *y, MKL_INT incy)
{
  dcopy(&n, x, &incx, y, &incy)
}

template <typename T, template<typename> class VectorX,
                      typename VectorY,
          Require< Dense<VectorX<T>>,
                   Dense<VectorY> > = true>
void
copy(const VectorX<T> &x, VectorY &&y)
{
    assert(x.length()==y.length());
    //hpc::ulmblas::copy(x.length(), x.conj(), x.data(), x.inc(), y.data(), y.inc());
    copy(x.length(), x.data(), x.inc(), y.data(), y.inc());
}

} } // namespace mklblas, hpc

#endif // HPC_MKLBLAS_MV_HPP
