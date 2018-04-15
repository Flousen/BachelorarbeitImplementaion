#ifndef HPC_MATVEC_QRTEST_HPP
#define HPC_MATVEC_QRTEST_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>


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

namespace hpc { namespace matvec {


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
  hpc::matvec::GeMatrix<TM> T(n, n);
  
  // appy H(1) *** H(n) to eye
  larft(A, tau, T);
  larfb(A, T, Q);
  
}

template <typename MatrixA, typename MatrixAqr, typename VectorTau,
          Require< Ge<MatrixA>, Ge<MatrixAqr>, Dense<VectorTau> > = true>
auto
qr_error(MatrixA &&A, MatrixAqr &&Aqr, VectorTau &&tau){

  hpc::matvec::GeMatrix<double> nA(A.numRows(), A.numCols());
  copy(A, nA);

  hpc::matvec::GeMatrix<double> Q(A.numRows(), A.numCols());
  makeQ(Aqr, tau, Q);
  
  std::size_t n = Aqr.numCols() ;
  hpc::matvec::GeMatrix<double> R(n, n);
  copy(Aqr.dim(n,n).view(UpLo::Upper), R);

  mm(1.0, Q,R,
     -1.0, nA);
  
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
