#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>
#include <hpc/matvec/test/diagdom.hpp>


// from intel mkl
#include <mkl_types.h>
#include <mkl_lapack.h>
#define MAX_M 5 
#define MAX_N 5


int
main()
{
    using namespace hpc::matvec;

    GeMatrix<double> A(MAX_M, MAX_N);
    DenseVector<double> x(A.numCols());

    test::rand(A);
    
    copy(A.row(0,0), x);

    auto B = A.view(UpLo::Upper);

    fmt::printf("B = \n"); print(B, "%9.4f");
    fmt::printf("x = \n"); print(x, "%9.4f");
    
    hpc::matvec::mv(1.0, B, x);    
//    hpc::mklblas::mv(1.0, B, x);    
    
    fmt::printf("x = \n"); print(x, "%9.4f");

}
