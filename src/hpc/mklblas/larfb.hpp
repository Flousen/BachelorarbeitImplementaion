#ifndef HPC_MKLBLAS_LARFB_HPP
#define HPC_MKLBLAS_LARFB_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>


namespace hpc { namespace mklblas {
void
larfb(  char side, char trans, char direct,
        char storev,  MKL_INT m, MKL_INT n,
        MKL_INT k, double* v, MKL_INT ldv,
        double* t, MKL_INT ldt, double* c,
        MKL_INT ldc, double* work, MKL_INT ldwork )
{
dlarfb( &side, &trans, &direct, &storev, &m, &n, &k,
        v, &ldv, t, &ldt,
        c, &ldc, work, &ldwork );
}

template <typename MatrixV, typename MatrixT, typename MatrixC>
void
larfb_ref(MatrixV &&V, MatrixT &&T, MatrixC &&C, bool trans = false)
{
  std::size_t m  = C.numRows();
  std::size_t n  = C.numCols();
  std::size_t k  = T.numCols();

  char tran = trans ? 'T' : 'N';

  hpc::matvec::GeMatrix<double> W(n,k, hpc::matvec::Order::ColMajor);

  larfb( 'L', tran, 'F', 'C', m, n, k,
        V.data(), V.incCol(), T.data(), T.incCol(),
        C.data(), C.incCol(), W.data(), W.incCol() );

}


//        hpc::mklblas::larft(A.block(i,i).dim(m-i,ib),
//               tau.block(i).dim(ib),
//               T.dim(ib,ib));
//        // Apply H' to A(i:m,i+ib:n) from the left
//        hpc::mklblas::larfb(A.block(i,i).dim(m-i,ib),
//              T.dim(ib,ib),
//              A.block(i,i + ib).dim(m-i, n-i-ib),
//              true);

} } // namespace mklblas, hpc


#endif // HPC_MKLBLAS_MV_HPP
