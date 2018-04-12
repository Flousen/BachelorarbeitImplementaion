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
#include <hpc/matvec/test/norminf.hpp>

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
    fmt::printf("%lf",i);
    print(A, "%9.4f");
    householderVector(A(i,i), A.col(i+1,i),tau(i));
    
    if (i < n && tau(i) != T(0)) {
      AII = A(i,i);
      A(i,i) = T(1);
      
      mv(T(1),
         A.block(i,i+1).view(Trans::view),
         A.col(i,i),
         T(0),
         work.block(i+1));

      rank1(-tau(i),
            A.col(i,i),
            work.block(i+1),
            A.block(i,i+1));

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
  std::size_t n  = V.numRows();

  if (n == 0){ return; }

  for (std::size_t i = 0; i< k; i++){
    if (tau(i) == 0){
      scal(TMV(0), T.col(0,i).dim(i));
    } else {
      auto VII = V(i,i);
      V(i,i) = TMV(1);
      
      // T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
      mv(-tau(i),
          V.block(i,0).dim(n-i,i).view(Trans::view),
          V.col(i,i),
          TMV(0),
          T.col(0,i).dim(i));

      V(i,i) = VII;
      
      // T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
      mv(TMV(1),
          T.block(0,0).dim(i,i).view(UpLo::Upper),
          T.col(0,i).dim(i) );
     
      T(i,i) = tau(i);
    }
  }
}

template <typename MatrixV, typename MatrixT, typename MatrixC>
void
larfb(MatrixV &&V, MatrixT &&T, MatrixC &&C, bool trans = false)
{
  std::size_t m  = C.numRows();
  std::size_t n  = C.numCols();
  std::size_t k  = T.numCols();

  trans = !trans;

  using TMV = ElementType<MatrixC>;
  GeMatrix<TMV> W(n,k);

  // W := C' * V  =  (C1'*V1 + C2'*V2)
  // W := C1'
  for(std::size_t j = 0; j < k; ++j){
    copy(C.row(j,0), W.col(0,j));
  }
  // W := W * V1
  mm(TMV(1),
      W,
      V.dim(k,k).view(UpLo::LowerUnit)
      );
  if(m > k){
    // W := W + C2'*V2
    mm(TMV(1),
        C.block(k,0).view(Trans::view),
        V.block(k,0),
        TMV(1),
        W);
  }
  // W :=  W * T
  mm(TMV(1),
     W,
     T.view(UpLo::Upper),
     trans);
// C := C - V * W'
  if(m > k){
    // C2 := C2 - V2 * W'
    mm(TMV(-1),
       V.block(k,0),
       W.view(Trans::view),
       TMV(1),
       C.block(k,0));
  }
  // W := W * V1'
  mm(TMV(1),
     W,
     V.dim(k,k).view(Trans::view).view(UpLo::UpperUnit));
  // C1 := C1 - W'
  for(std::size_t j = 0; j < k; j++){
    for(std::size_t i = 0; i < n; i++){
      C(j,i) -= W(i,j);
    } 
  }

}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_blk(MatrixA &&A, VectorTau &&tau)
{
  using TMA = ElementType<MatrixA>;

  std::size_t m  = A.numRows();
  std::size_t n  = A.numCols();
  std::size_t mn = std::min(m,n);

  assert(tau.length() == mn);
  std::size_t nb = 5 ; 
  std::size_t nx = 5 ;
  std::size_t nbmin = 2 ;

  GeMatrix<TMA> T(nb, nb);
  std::size_t i = 1;
  if(nb >= nbmin && nb < mn && nx < mn){
    for (i = 0; i < mn-nx; i+=nb){
      std::size_t ib = std::min(mn-i+1, nb);
      qr_unblk(A.block(i,i).dim(m-i,ib), tau.block(i).dim(ib));
     
      
      if ( i + ib <= n){
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1)

        larft(A.block(i,i).dim(m-i,ib),
               tau.block(i).dim(ib),
               T.dim(ib,ib));
        // Apply H' to A(i:m,i+ib:n) from the left
        larfb(A.block(i,i).dim(m-i,ib),
              T.dim(ib,ib),
              A.block(i,i + ib).dim(m-i, n-i-ib),
              true);

      }
    }
  }
  if ( i <= mn){
    qr_unblk(A.block(i,i).dim(m-i,n-i), tau.block(i).dim(m-i));
  }
}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_blke(MatrixA &&A, VectorTau &&tau)
{
  using TMA = ElementType<MatrixA>;

  std::size_t m  = A.numRows();
  std::size_t n  = A.numCols();
  std::size_t mn = std::min(m,n);

  assert(tau.length() == mn);
  std::size_t nb = 5 ; 

  GeMatrix<TMA> T(nb, nb);
  std::size_t i = 1;
  if( nb < mn ){
    for (i = 0; i < mn; i+=nb){
      //nb = std::min(mn-i, nb)
      qr_unblk(A.block(i,i).dim(m-i,nb), tau.block(i).dim(nb));
      if ( i + nb <= n){
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+nb-1)

        larft(A.block(i,i).dim(m-i,nb),
               tau.block(i).dim(nb),
               T.dim(nb,nb));
        // Apply H' to A(i:m,i+nb:n) from the left
        larfb(A.block(i,i).dim(m-i,nb),
              T.dim(nb,nb),
              A.block(i,i + nb).dim(m-i, n-i-nb),
              true);
      }
    }
  }
  if ( i <= mn ){
    qr_unblk(A.block(i,i).dim(m-i,n-i), tau.block(i).dim(m-i));
  }
}

template <typename MatrixA, typename VectorTau,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void
qr_blk2(MatrixA &&A, VectorTau &&tau)
{
  using TMA = ElementType<MatrixA>;

  std::size_t m  = A.numRows();
  std::size_t n  = A.numCols();
  std::size_t mn = std::min(m,n);

  assert(tau.length() == mn);
  std::size_t nb = 5 ; 

  GeMatrix<TMA> T(nb, nb);

  if( nb < mn ){
    //nb = std::min(mn-i+1, nb)
    qr_unblk(A.dim(m,nb), tau.dim(nb));

    // Form the triangular factor of the block reflector
    
    // H = H(i) H(i+1) . . . H(i+nb-1)
    larft(A.dim(m,nb),
           tau.dim(nb),
           T);
    
    // Apply H' to A(i:m,i+nb:n) from the left
    larfb(A.dim(m,nb),
          T,
          A.block(0,nb).dim(m, n-nb),
          true);

    qr_blk2(A.block(nb,nb), tau.block(nb));
  }
  else if ( nb <= mn ){
    qr_unblk(A, tau);
  }
}

template <typename MatrixA, typename VectorTau, typename MatrixQ,
          Require< Ge<MatrixA>, Dense<VectorTau> > = true>
void makeQ(MatrixA &&A, VectorTau &&tau, MatrixQ &&Q){
  assert(A.numRows() == Q.numRows());
  assert(A.numCols() == Q.numCols());
  

  // make eye
  std::size_t n  = A.numCols();
  std::size_t m  = A.numRows();
  for(std::size_t i = 0; i < n; ++i){
    for(std::size_t j = 0; j < m; ++j){
      Q(j,i) = (i==j) ? 1 : 0;
    }
  }
  
  using TM = ElementType<MatrixA>;
  GeMatrix<TM> T(n, n);
  
  // appy H(1) *** H(n) to eye
  larft(A, tau, T);
  larfb(A, T, Q);
  
}

template <typename MatrixA, typename MatrixAqr, typename VectorTau,
          Require< Ge<MatrixA>, Ge<MatrixAqr>, Dense<VectorTau> > = true>
auto
qr_error(MatrixA &&A, MatrixAqr &&Aqr, VectorTau &&tau){

  GeMatrix<double> nA(A.numRows(), A.numCols());
  copy(A, nA);

  GeMatrix<double> Q(A.numRows(), A.numCols());
  makeQ(Aqr, tau, Q);
  
  std::size_t n = Aqr.numCols() ;
  GeMatrix<double> R(n, n);
  
  copy(Aqr.dim(n,n).view(UpLo::Upper), R);
  
  auto normAn = test::norminf(nA);
  auto normA  = test::norminf( A);
  auto eps = std::numeric_limits<Real<double>>::epsilon();
  auto err = normAn / (normA *
      std::min(A.numRows(), A.numCols()) * 
      eps);
  return err;
  
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_QR_HPP
