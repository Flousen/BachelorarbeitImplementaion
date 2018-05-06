#ifndef HPC_MKLBLAS_QRFLO_HPP
#define HPC_MKLBLAS_QRFLO_HPP

#include <cstddef>
#include <hpc/assert.hpp>
#include <hpc/matvec/traits.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_blas.h>


namespace hpc { namespace mklblas {

template <typename VectorV, typename Alpha, typename Tau,
          Require< Dense<VectorV> > = true>
void
householderVector(Alpha &alpha, VectorV &&v, Tau &tau)
{
  if (v.length() == 0) {
    tau = Tau(0);
    return;
  }
  //T xnorm = norm();
  //auto dot = hpc::ulmblas::dot( v.length(),
  //            false, v.data(), v.inc(),
  //            false, v.data(), v.inc() );

  double dotprod = hpc::mklblas::dot(v,v);
  //double dotprod = hpc::matvec::dot(v,v);
  if (dotprod == ElementType<VectorV>(0)){
    tau = Tau(0);
  } else {
    auto beta = -std::copysign(sqrt(alpha*alpha + dotprod),alpha);
    tau = (beta - alpha) / beta;
    hpc::mklblas::scal( 1/(alpha - beta), v);
    //hpc::matvec::scal( 1/(alpha - beta), v);
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
  std::size_t mn  = std::min(m,n);
  assert(tau.length() == n);
  
  hpc::matvec::DenseVector<T> W(mn);
  for (std::size_t i = 0; i < n; ++i){
    hpc::mklblas::householderVector(A(i,i), A.col(i+1,i),tau(i));
    //hpc::matvec::householderVector(A(i,i), A.col(i+1,i),tau(i));
    
    if (i < n && tau(i) != T(0)) {
      AII = A(i,i);
      A(i,i) = T(1);
      
      //hpc::mklblas::mv(T(1),
      hpc::matvec::mv(T(1),
        A.block(i,i+1).view(hpc::matvec::Trans::view),
        A.col(i,i),
        T(0),
        W.block(i+1));

      hpc::mklblas::rank1(-tau(i),
      //hpc::matvec::rank1(-tau(i),
        A.col(i,i),
        W.block(i+1),
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
      hpc::mklblas::scal(TMV(0), T.col(0,i).dim(i));
    } else {
      auto VII = V(i,i);
      V(i,i) = TMV(1);
      
      // T(1:i-1,i) := - tau(i) * V(i:n,1:i-1)' * V(i:n,i)
      //hpc::mklblas::mv(-tau(i),
      hpc::matvec::mv(-tau(i),
          V.block(i,0).dim(n-i,i).view(hpc::matvec::Trans::view),
          V.col(i,i),
          TMV(0),
          T.col(0,i).dim(i));

      V(i,i) = VII;
      
      // T(1:i-1,i) := T(1:i-1,1:i-1) * T(1:i-1,i)
      hpc::mklblas::mv(TMV(1),
      //hpc::matvec::mv(TMV(1),
          T.block(0,0).dim(i,i).view(hpc::matvec::UpLo::Upper),
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
  hpc::matvec::GeMatrix<TMV> W(n,k);

  // W := C' * V  =  (C1'*V1 + C2'*V2)
  // W := C1'
  for(std::size_t j = 0; j < k; ++j){
    hpc::mklblas::copy(C.row(j,0), W.col(0,j));
  }
  // W := W * V1
  //hpc::mklblas::mm(TMV(1),
  hpc::matvec::mm(TMV(1),
      W,
      V.dim(k,k).view(hpc::matvec::UpLo::LowerUnit)
      );
  if(m > k){
    // W := W + C2'*V2
    //hpc::mklblas::mm(TMV(1),
    hpc::matvec::mm(TMV(1),
        C.block(k,0).view(hpc::matvec::Trans::view),
        V.block(k,0),
        TMV(1),
        W);
  }
  // W :=  W * T
  //hpc::mklblas::mm(TMV(1),
  hpc::matvec::mm(TMV(1),
     W,
     T.view(hpc::matvec::UpLo::Upper),
     trans);
// C := C - V * W'
  if(m > k){
    // C2 := C2 - V2 * W'
    //hpc::mklblas::mm(TMV(-1),
    hpc::matvec::mm(TMV(-1),
       V.block(k,0),
       W.view(hpc::matvec::Trans::view),
       TMV(1),
       C.block(k,0));
  }
  // W := W * V1'
  //hpc::mklblas::mm(TMV(1),
  hpc::matvec::mm(TMV(1),
     W,
     V.dim(k,k).view(hpc::matvec::Trans::view).view(hpc::matvec::UpLo::UpperUnit));
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

  hpc::matvec::GeMatrix<TMA> T(nb, nb);
  std::size_t i = 1;
  if(nb >= nbmin && nb < mn && nx < mn){
    for (i = 0; i < mn-nx; i+=nb){
      std::size_t ib = std::min(mn-i+1, nb);
      //hpc::mklblas::qr_unblk_ref(A.block(i,i).dim(m-i,ib), tau.block(i).dim(ib));
      hpc::mklblas::qr_unblk(A.block(i,i).dim(m-i,ib), tau.block(i).dim(ib));

      if ( i + ib <= n){
        // Form the triangular factor of the block reflector
        // H = H(i) H(i+1) . . . H(i+ib-1)

        hpc::mklblas::larft(A.block(i,i).dim(m-i,ib),
               tau.block(i).dim(ib),
               T.dim(ib,ib));
        // Apply H' to A(i:m,i+ib:n) from the left
        hpc::mklblas::larfb(A.block(i,i).dim(m-i,ib),
              T.dim(ib,ib),
              A.block(i,i + ib).dim(m-i, n-i-ib),
              true);

      }
    }
  }
  if ( i <= mn){
    hpc::mklblas::qr_unblk_ref(A.block(i,i).dim(m-i,n-i), tau.block(i).dim(n-i));
  }
  fmt::printf("schaff ich es hier her?\n");
}

/*

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

  hpc::matvec::GeMatrix<TMA> T(nb, nb);
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
    qr_unblk(A.block(i,i).dim(m-i,n-i), tau.block(i).dim(n-i));
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

  hpc::matvec::GeMatrix<TMA> T(nb, nb);

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
*/


} } // namespace mklblas, hpc


#endif // HPC_MKLBLAS_MV_HPP
