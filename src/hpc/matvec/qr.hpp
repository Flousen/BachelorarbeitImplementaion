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

template <typename VectorV, typename Alpha, typename Tau>
void
householderVector(Alpha &alpha, VectorV &&v, Tau &tau)
{
  print(v);

  if (v.length() == 0) {
    tau = Tau(0);
    return;
  }

  //T xnorm = norm();
  auto dot = hpc::ulmblas::dot( v.length(),
              false, v.data(), v.inc(),
              false, v.data(), v.inc() );
  if (dot == ElementType<VectorV>(0)){
    tau = Tau(0);
  } else {
    auto beta = -std::copysign(sqrt(alpha*alpha + dot),alpha);
    tau = (beta - alpha) / beta;
    scal(1/(alpha - beta),v);
    alpha = beta;
  }
  
}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_unblk(MatrixA &&A, VectorTau &&tau)
{
  using T = ElementType<MatrixA>;
  T AII = 0;
  std::size_t m  = A.numRows();
  std::size_t n  = A.numCols();
  std::size_t mn = std::min(m,n);

  assert(tau.length() == mn);

  DenseVector<T> work(mn);
  for (std::size_t i = 0; i < mn; ++i){
    householderVector(A(i,i), A.col(i+1,i),tau(i));
    if (i < n && tau(i) != T(0)) {
      AII = A(i,i);
      A(i,i) = T(1);
      
      mv(T(1), A.block(i,i+1).view(Trans::view), A.col(i,i), T(0), work.block(i+1));
      rank1(-tau(i), A.col(i,i), work.block(i+1), A.block(i,i+1));

      A(i,i) = AII;
    }
    fmt::printf("i=%d\n", i);
    print(A);
  }
}

} } // namespace matvec, hpc


#endif // HPC_MATVEC_QR_HPP
