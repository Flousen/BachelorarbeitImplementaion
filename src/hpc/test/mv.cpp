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

    GeMatrix<double> A(MAX_M, MAX_N, Order::ColMajor);
    DenseVector<double> x(A.numCols());
    DenseVector<double> y(A.numCols());
    
    GeMatrix<double> A1(MAX_M, MAX_N, Order::ColMajor);
    DenseVector<double> x1(A.numCols());
    DenseVector<double> y1(A.numCols());

    test::rand(A);
    copy(A.row(0,0), x);
    copy(A.row(0,0), y);

    copy(A, A1);
    copy(x, x1);
    copy(y, y1);


    fmt::printf("A = \n"); print(A, "%9.4f");
    fmt::printf("x = \n"); print(x, "%9.4f");
    fmt::printf("y = \n"); print(y, "%9.4f");
    
    fmt::printf("A1 = \n"); print(A1, "%9.4f");
    fmt::printf("x1 = \n"); print(x1, "%9.4f");
    fmt::printf("y1 = \n"); print(y1, "%9.4f");
    
    
    hpc::matvec::mv (1.0, A.view(hpc::matvec::Trans::view), x, 1.0, y);    
    hpc::mklblas::mv(1.0, A1.view(hpc::matvec::Trans::view), x1, 1.0, y1);    
    
    fmt::printf("matvec = \n"); print(y, "%9.4f");
    fmt::printf("mkl = \n"); print(y1, "%9.4f");

    fmt::printf("incRow = %d\n",A.incRow());
    fmt::printf("incRow = %d\n",A.view(hpc::matvec::Trans::view).incRow());
    fmt::printf("incCol = %d\n",A.incCol());
    fmt::printf("incCol = %d\n",A.view(hpc::matvec::Trans::view).incCol());

}
