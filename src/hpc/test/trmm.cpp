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
    GeMatrix<double> B(MAX_M, MAX_N, Order::ColMajor);
    GeMatrix<double> C(MAX_M, MAX_N, Order::ColMajor);
    test::rand(A);
    test::rand(B);
    copy(B,C);

    auto T = A.view(UpLo::Upper);

    fmt::printf("T = \n"); print(T, "%9.4f");
    fmt::printf("B = \n"); print(B, "%9.4f");
    fmt::printf("C = \n"); print(C, "%9.4f");
    
    //hpc::matvec::mm (1.0, T, B);    
    hpc::mklblas::mm(1.0, T, C);    
    
    fmt::printf("matvec = \n"); print(B, "%9.4f");
    fmt::printf("mkl    = \n"); print(C, "%9.4f");

}
