#ifndef HPC_MATVEC_QR_HPP
#define HPC_MATVEC_QR_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/ulmblas/dot.hpp>

#include <hpc/matvec/iamax.hpp>
#include <hpc/matvec/rank1.hpp>
#include <hpc/matvec/scal.hpp>
#include <hpc/matvec/sv.hpp>
#include <hpc/matvec/sm.hpp>
#include <hpc/matvec/swap.hpp>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/axpy.hpp>

namespace hpc { namespace matvec {

template <typename VectorV, typename A, typename T>
void
householderVector(VectorV &&v, A *alpha, T *tau)
{

  if (v.length() == 1)
    tau = T(0);

  //T xnorm = norm();
  auto dot = hpc::ulmblas::dot( v.length(),
              false, v.data(), v.inc(),
              false, v.data(), v.inc() );

  if (dot == VectorV(0)){
    tau = T(0);
  } else {
    auto beta = -std::copysign(sqrt(alpha*alpha + dot),alpha);
    tau = (beta - alpha) / beta;
    scal(1/(alpha - beta),v);
    alpha = beta;
  }
  
}

template <typename MatrixA,
          Require< Ge<MatrixA> > = true>
std::ptrdiff_t
qr_unblk(MatrixA &&A)
{
  using T = ElementType<MatrixA>;
  T AII = 0;
  std::size_t m  = A.numRows();
  std::size_t n  = A.numCols();
  std::size_t mn = std::min(m,n);

  DenseVector<T> tau(mn);

  for (std::size_t i = 0; i < mn; ++i){
    //householderVector(view_select(A.row(i,m-i)),A(i,i),tau(i));
    if (i < n && tau(i) != 0) {
      AII = A(i,i);
      A(i,i) = T(1);
      
      DenseVector<T> work(mn-i);
      mv(T(1),A(i,i),A.row(i,i),T(0), work);
      rank1(tau(i),work,A.row(i,i),A(i,i+1));

      A(i,i) = AII;
    }
  }
}

} } // namespace matvec, hpc


#endif // HPC_MATVEC_QR_HPP
