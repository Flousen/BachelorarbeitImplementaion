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
void qr_unblk(MatrixA &&A, VectorTau &&tau)
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
  }
}

template <typename MatrixV, typename VectorTau, typename MatrixT,
          Require< Ge<MatrixV>, Dense<VectorTau>, Ge<MatrixT> > = true>
void
LARFT(MatrixV &&V, VectorTau &&tau, MatrixT &&T)
{
  std::size_t n  = A.numCols();
  std::size_t k  = A.numRows();
  assert();
  for (std::size_t i = 0; i< k; i++){
    if (tau(i) == 0){
      scal(0,T.row(1,i).dim(i,i));
    } else {
      auto VII = V(i,i);
      V(i,i) = 0;
      // T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
      mv(-tau(i), V.block(i,1).dim(i,n-i).view(Trans::view), V.row(i,i), 0, T.col(1,i).dim(i));
      V(i,i) = VII;
      // T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
      trmv(T.block(1,1).dim(i,i),T.col(1,i).dim(i));
      T(i,i) = tau(i);
    }
  }
}

template <typename MatrixV, typename MatrixT, typename MatrixC,
          Require< Ge<MatrixV>, Ge<MatrixT>, Ge<MatrixC> > = true>
void
LARFB(MatrixV &&V, MatrixT &&T, MatrixC &&C)
{
  std::size_t m  = C.numRows();
  std::size_t n  = C.numCols();
  std::size_t k  = A.numRows();
  GeMatirx<T> W(m,n);

  copy(C,W);
  trmm(1,V,W);
  if (m>k){
    mm(1,C,V,W);
  }
  trmm(1,T,W);
  if (m>k){
    mm(-1,V,W,C);
  }
  trmm(1,V,W);
  C += -W;
}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_blk(MatrixA &&A, VectorTau &&tau)
{

  using T = ElementType<MatrixA>;

  std::size_t m  = A.numRows();
  std::size_t n  = A.numCols();
  std::size_t mn = std::min(m,n);

  assert(tau.length() == mn);
  std::size_t nb; 
  std::size_t ib;
  GeMatirx<T> H(m,n);
  if(nb >= nbmin && nb < mn && nx < mn){
    for (std::size_t i = 0; i < mn-nx; i+=nb){
      //ib = std::min(mn-i+1, nb)
      qr_unblk(A.block(i,i).block(m-i+1,ib), tau.block(i).dim(m-i));
      if ( i + ib <= n){
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1)
        LARFT(A.block(i,i).dim(ib), tau.block(i).dim(ib), H);
        // Apply H' to A(i:m,i+ib:n) from the left
        LARFB(A.block(i,i).dim(ib), H, A.block(i,i + ib).dim(n-i-ib));
      }
    }
  } else {
    std::size_t i = 0;
  }
  if ( i <= mn)
    qr_unblk(A.block(i,i).dim(m-i,n-i), tau.block(i).dim(m-i));
}

} } // namespace matvec, hpc


#endif // HPC_MATVEC_QR_HPP
