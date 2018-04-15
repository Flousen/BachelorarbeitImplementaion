#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>

#define MAX_M 20 
#define MAX_N 10 

int
main()
{
  
  //auto qr= <GeMatrix<double> &, DenseVector<std::size_t> &>; 

  hpc::matvec::GeMatrix<double> A(MAX_M, MAX_N, hpc::matvec::Order::ColMajor);
  hpc::matvec::DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));

  hpc::matvec::GeMatrix<double> B(MAX_M, MAX_N, hpc::matvec::Order::ColMajor);
  hpc::matvec::DenseVector<double> tauB(std::min(A.numRows(),A.numCols()));

  hpc::matvec::test::rand(A);
  hpc::matvec::copy(A,B);

  hpc::mklblas::scal(2.0, tauA);

  fmt::printf("A = \n");
  hpc::matvec::print(A, "%9.4f");

  //qr_unblk(A,tauA);
  //hpc::mklblas::qr_blk_ref(A, tauA);
  //hpc::mklblas::qr_unblk(A, tauA);
  //hpc::mklblas::qr_blk(A, tauA);
  //hpc::matvec::qr_unblk(A, tauA);
  hpc::matvec::qr_blke(A, tauA);

  fmt::printf("tauA = \n");
  hpc::matvec::print(tauA);
  fmt::printf("qr(A) = \n");
  hpc::matvec::print(A, "%9.4f");
  
  double err = hpc::matvec::qr_error(B,A,tauA);
  fmt::printf("m = %lf\n", A.numRows());
  fmt::printf("n = %lf\n", A.numCols());
  fmt::printf("err = %lf\n", err);

}
