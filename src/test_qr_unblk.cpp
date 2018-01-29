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
#define MAX_M 2000
#define MAX_N 2000

int
main()
{
  using namespace hpc::matvec;

  GeMatrix<double> A(MAX_M, MAX_N);

  test::rand(A);
  qr_unblk(A);

}
