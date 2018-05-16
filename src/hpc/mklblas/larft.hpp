#ifndef HPC_MKLBLAS_LARFT_HPP
#define HPC_MKLBLAS_LARFT_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>


namespace hpc { namespace mklblas {

void
larft( char direct, char storev, MKL_INT n,
             MKL_INT k, double* v, MKL_INT ldv,
             const double* tau, double* t, MKL_INT ldt )
{
  dlarft(&direct, &storev, &n, &k,
         v, &ldv, tau, t, &ldt );
}


// H  =  I - V * T * V'
template <typename MatrixV, typename VectorTau, typename MatrixT>
void
larft_ref(MatrixV &&V, VectorTau &&tau, MatrixT &&T)
{

  std::size_t k  = tau.length();
  std::size_t n  = V.numRows();
  larft('F', 'C', n, k,
        V.data(), V.incCol(),
        tau.data(), T.data(), T.incCol() );

}

} } // namespace mklblas, hpc


#endif // HPC_MKLBLAS_MV_HPP
