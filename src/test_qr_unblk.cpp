#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/trmatrix.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>
#include <hpc/matvec/qr.hpp>
#include <hpc/matvec/mm.hpp>
#include <hpc/matvec/print.hpp>

#define MIN_M 10
#define MIN_N 10
#define INC_M 10
#define INC_N 10
#define MAX_M 10 
#define MAX_N 10 

int
main()
{
  using namespace hpc::matvec;

  GeMatrix<double> A(MAX_M, MAX_N);
  DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));
  GeMatrix<double> B(MAX_M, MAX_N);
  DenseVector<double> tauB(std::min(A.numRows(),A.numCols()));
  GeMatrix<double> Ares(MAX_M, MAX_N);

  test::rand(A);
  copy(A,B);

  fmt::printf("A = \n");
  print(A, "%9.4f");

  qr_unblk(A,tauA);
  print(tauA);
  fmt::printf("qr(A) = \n");
  print(A, "%9.4f");
  
  qr_blk(B,tauB);
  print(tauB);
  fmt::printf("qr_blk(A) = \n");
  print(B, "%9.4f");

  //larft(A,tau,B);
  //larfb(A,B,A);

  //fmt::printf("A =\n");
  //print(A, "%9.4f");
  
  //qr_blk(A,tau);
  //print(tau);
  //fmt::printf("qr_blk(A) = \n");
  //print(A, "%9.4f");

  //makeQ(A, tau, Ares);
  //fmt::printf("Ares = \n");
  //print(Ares, "%9.4f");

}
