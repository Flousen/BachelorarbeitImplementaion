#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>
#include <hpc/matvec/test/diagdom.hpp>

#include <hpc/mklblas.hpp>


// from intel mkl
#include <mkl_types.h>
#include <mkl_lapack.h>
#define MAX_M 5 
#define MAX_N 5


int
main()
{
  using namespace hpc::matvec;
  //DenseVector<double> x(A.numCols());

  GeMatrix<double> A(5,5); 
 // test::rand(A);
  test::init(A);
  fmt::printf("A = \n"); print(A, "%9.4f");
  fmt::printf("A = \n"); print(A.block(2,2), "%9.4f");
  fmt::printf("A = \n"); print(A.block(2,2).view(Trans::view), "%9.4f");

  return 0;   
}
