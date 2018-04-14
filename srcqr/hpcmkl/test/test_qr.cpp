#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpcmkl/matvec.hpp>
#include <hpcmkl/matvec/test/error.hpp>
#include <hpcmkl/matvec/test/rand.hpp>
#include <hpcmkl/matvec/test/walltime.hpp>
#include <hpcmkl/matvec/test/diagdom.hpp>

// from intel mkl
#include <mkl_types.h>
#include <mkl_lapack.h>

#define MAX_M 20
#define MAX_N 10

int
main()
{
  using namespace hpc::matvec;

  auto qr = qr_unblk<GeMatrix<double> &, DenseVector<std::size_t> &>; 

  GeMatrix<double> A(MAX_M, MAX_N);
  DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));

  GeMatrix<double> B(MAX_M, MAX_N);
  DenseVector<double> tauB(std::min(A.numRows(),A.numCols()));

  test::rand(A);
  copy(A,B);

  fmt::printf("A = \n");
  print(A, "%9.4f");

  qr(A,tauA);

  fmt::printf("tauA = \n");
  print(tauA);
  fmt::printf("qr(A) = \n");
  print(A, "%9.4f");
  
  double err = qr_error(B,A,tauA);
  fmt::printf("m = %lf\n", A.numRows());
  fmt::printf("n = %lf\n", A.numCols());
  fmt::printf("err = %lf\n", err);

}