#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>

#define MAX_M 200
#define MAX_N 100

int
main()
{
  //using namespace hpc::matvec;
  
  //auto qr= <GeMatrix<double> &, DenseVector<std::size_t> &>; 

  hpc::matvec::GeMatrix<double> A(MAX_M, MAX_N, hpc::matvec::Order::ColMajor);
  hpc::matvec::DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));

  hpc::matvec::GeMatrix<double> B(MAX_M, MAX_N, hpc::matvec::Order::ColMajor);
  hpc::matvec::DenseVector<double> tauB(std::min(A.numRows(),A.numCols()));

  hpc::matvec::test::rand(A);
  hpc::matvec::copy(A,B);

  //fmt::printf("A = \n");
  //print(A, "%9.4f");
  //fmt::printf("tauA = \n");
  //print(tauA);

  //qr_unblk(A,tauA);
  //hpc::mklblas::qr_blk_ref(A, tauA);
  hpc::mklblas::qr_blk(A, tauA);

  //fmt::printf("tauA = \n");
  //print(tauA);
  //fmt::printf("qr(A) = \n");
  //print(A, "%9.4f");
  
  double err = hpc::matvec::qr_error(B,A,tauA);
  fmt::printf("m = %lf\n", A.numRows());
  fmt::printf("n = %lf\n", A.numCols());
  fmt::printf("err = %lf\n", err);

}
