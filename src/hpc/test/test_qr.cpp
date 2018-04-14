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
  using namespace hpc::matvec;
  
  //auto qr= <GeMatrix<double> &, DenseVector<std::size_t> &>; 

  GeMatrix<double> A(MAX_M, MAX_N, Order::ColMajor);
  DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));

  GeMatrix<double> B(MAX_M, MAX_N, Order::ColMajor);
  DenseVector<double> tauB(std::min(A.numRows(),A.numCols()));

  test::rand(A);
  copy(A,B);

  fmt::printf("A = \n");
  print(A, "%9.4f");

  //qr_unblk(A,tauA);
  hpc::mklblas::qr_unblk(A,tauA);

  fmt::printf("tauA = \n");
  print(tauA);
  fmt::printf("qr(A) = \n");
  print(A, "%9.4f");
  
  double err = qr_error(B,A,tauA);
  fmt::printf("m = %lf\n", A.numRows());
  fmt::printf("n = %lf\n", A.numCols());
  fmt::printf("err = %lf\n", err);

}
