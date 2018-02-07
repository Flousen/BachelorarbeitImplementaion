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

#define MIN_M 10
#define MIN_N 10
#define INC_M 10
#define INC_N 10
#define MAX_M 5 
#define MAX_N 5 

int
main()
{
  using namespace hpc::matvec;

  GeMatrix<double> A(MAX_M, MAX_N);
  GeMatrix<double> B(MAX_M, MAX_N);
  DenseVector<double> tau(std::min(A.numRows(),A.numCols()));

  //test::rand(A);
  //print(A, "%9.4f");
  //qr_unblk(A,tau);
  //print(A, "%9.4f");
  //print(tau);

  test::rand(A);
  test::rand(B);
  fmt::printf("A =\n");
  print(A);
  fmt::printf("B =\n");
  print(B);

  
  qr_blk(A,tau);
  fmt::printf("qr_blk(A) = \n");
  print(A, "%9.4f");

}
