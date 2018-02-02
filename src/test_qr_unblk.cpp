#include <complex>
#include <functional>
#include <random>


#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>
#include <hpc/matvec/qr.hpp>

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
  DenseVector<double> tau(std::min(A.numRows(),A.numCols()));

  test::rand(A);
  print(A, "%9.4f");
  qr_unblk(A,tau);
  print(A, "%9.4f");
  print(tau);

  print(A+A);

}
