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
#include <hpc/matvec/copy.hpp>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/gematrix.hpp>
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


// H  =  I - V * T * V'
template <typename MatrixV, typename VectorTau, typename MatrixT>
void
larft(MatrixV &&V, VectorTau &&tau, MatrixT &&T)
{

  using TMV = ElementType<MatrixV>;
  std::size_t k  = tau.length();
  std::size_t n  = V.numCols();
  if (n == 0)
    return;

  for (std::size_t i = 0; i< k; i++){
    if (tau(i) == 0){
      scal(TMV(0), T.row(1,i).dim(i));
    } else {
      auto VII = V(i,i);
      V(i,i) = TMV(0);

      // T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
      mv(-tau(i), V.block(i,0).dim(n-i,i).view(Trans::view),
          V.row(i,i), TMV(0),
          T.col(0,i).dim(i));
      V(i,i) = VII;
    //T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
      mv(TMV(1), T.block(0,0).dim(i,i),
          T.col(0,i).dim(i), TMV(0),
          T.col(0,i).dim(i));
      T(i,i) = tau(i);
    }
  }
}

template <typename MatrixV, typename MatrixT, typename MatrixC>
void
larfb(MatrixV &&V, MatrixT &&T, MatrixC &&C)
{
  std::size_t m  = C.numRows();
  std::size_t n  = C.numCols();
  std::size_t k  = V.numCols();

  fmt::printf("m  = %lf \n",m);
  fmt::printf("n  = %lf \n",n);
  fmt::printf("k  = %lf \n",k);

  using TMV = ElementType<MatrixC>;
  GeMatrix<TMV> W(T.numRows(),k);
  fmt::printf("V rows = %lf V cols = %lf\n",V.numRows(),V.numCols());
  fmt::printf("T rows = %lf T cols = %lf\n",T.numRows(),T.numCols());
  fmt::printf("C rows = %lf C cols = %lf\n",C.numRows(),C.numCols());
  fmt::printf("W rows = %lf W cols = %lf\n",W.numRows(),W.numCols());
  
  // W <- 1*C1'V1 + 0*W
  mm(TMV(1), 
      C.dim(k,n).view(Trans::view).view(UpLo::LowerUnit),
      V.dim(n,k),
      TMV(0),
      W.dim(k,k));
  
  if ( m > k ){
    //   W <- C2'*V2 + W
    mm()
  }

  // W := W * T'  or  W * T
  mm();

  
  if ( m > k ){
    // C2 <- -1*V2*W' + C2
    mm();
  }

  // C1 <- -1*V1*W' + 1*C1
  mm();


  // oder so 
  // W <- 1*C1'V1 + 0*W
  // W <- 1*C2'*V2 + 1*W
  // W <- W*T
  // C2 <- -1*V2*W' + 1*C
  // C1 <- -1*V1*W' + 1*C1

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
  std::size_t nb = 5 ; 
  std::size_t nx = 5 ;
  std::size_t nbmin = 2 ;

  std::size_t ib = 5 ;
  GeMatrix<T> H(n, n);
  std::size_t i = 1;
  if(nb >= nbmin && nb < mn && nx < mn){
    for (i = 0; i < mn-nx; i+=nb){
      //ib = std::min(mn-i+1, nb)
      qr_unblk(A.block(i,i).dim(m-i,ib), tau.block(i).dim(ib));
      if ( i + ib <= n){
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1)
        larft(A.block(i,i).dim(m-1,ib), tau.block(i).dim(ib),
               H.block(0,0).dim(n,ib));
        fmt::printf("H =\n");
        print(H);
        // Apply H' to A(i:m,i+ib:n) from the left
        larfb(A.block(i,i).dim(m,ib),
              H.block(0,0).dim(n,ib),
              A.block(i,i + ib).dim(m-i, n-i-ib));
      }
    }
  }
  if ( i <= mn)
    qr_unblk(A.block(i,i).dim(m-i,n-i), tau.block(i).dim(m-i));
}

} } // namespace matvec, hpc


#endif // HPC_MATVEC_QR_HPP
