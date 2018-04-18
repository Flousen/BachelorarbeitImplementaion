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
    
    GeMatrix<double> A1(MAX_M, MAX_N, Order::ColMajor);
    DenseVector<double> x1(A.numCols());

    test::rand(A);
    copy(A.row(0,0), x);

    copy(A, A1);
    copy(x, x1);


    fmt::printf("A = \n"); print(A, "%9.4f");
    fmt::printf("x = \n"); print(x, "%9.4f");
    
    fmt::printf("A1 = \n"); print(A1, "%9.4f");
    fmt::printf("x1 = \n"); print(x1, "%9.4f");
    
    
        
       
    
    fmt::printf("matvec = %lf\n", hpc::matvec::dot (x, x));
    fmt::printf("mkl = %lf\n",hpc::mklblas::dot(x1, x1) ); 

}
