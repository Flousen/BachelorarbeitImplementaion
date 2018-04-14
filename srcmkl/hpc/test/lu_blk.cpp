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

namespace hpc { namespace matvec {

template <typename T, template<typename> class MatrixA,
          typename lu_func,
          Require< Ge<MatrixA<T>> > = true>
std::pair<double, double>
test_lu(const MatrixA<T> &A0, lu_func lu)
{
    GeMatrix<double>         A(A0.numRows(), A0.numCols(), Order::ColMajor);
    DenseVector<std::size_t> piv(A.numRows());

    copy(A0, A);
    std::ptrdiff_t res;

    test::WallTime<double> timer;

    timer.tic();
    res = lu(A, piv);
    double time = timer.toc();
    double err  = test::error_estimate::getrf(A0, A, piv);

    if (res!=-1) {
        fmt::printf("Matrix is (numerically) singular\n");
    }
    return std::pair<double, double>(err, time);
}

template <typename T, template<typename> class MatrixA,
          Require<Ge<MatrixA<T>>> = true>
std::pair<double, double>
test_mkl_lu(const MatrixA<T> &A0)
{
    auto M = A0.numRows();
    auto N = A0.numCols();

    matvec::GeMatrix<double>         A_(M, N, matvec::Order::ColMajor);
    matvec::DenseVector<std::size_t> piv(M);

    auto A = A_.dim(M,N);

    copy(A0, A);
    MKL_INT res;

    test::WallTime<double> timer;

    MKL_INT m   = A.numRows();
    MKL_INT n   = A.numCols();
    MKL_INT lda = A.incCol(); assert(A.incRow()==1);

    timer.tic();
    dgetrf(&m, &n, A.data(), &lda, (MKL_INT*)piv.data(), &res);
    double time = timer.toc();

    for (std::size_t i=0; i<piv.length(); ++i) {
        --piv(i);
    }

    double err  = test::error_estimate::getrf(A0, A, piv);

    if (res!=0) {
        fmt::printf("Matrix is (numerically) singular\n");
    }
    return std::pair<double, double>(err, time);

}


} } // namespace matvec, hpc

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
    test::diagdom(A);

    // select here the LU variant for testing
    auto lu = lu_blk_var1<GeMatrix<double> &, DenseVector<std::size_t> &>;

    fmt::printf("%5s %5s "
                "%15s %15s %15s "
                "%15s %15s %15s %15s\n",
                "M", "N",
                "Error 1", "MKL (Time 1)", "MFLOPS 1",
                "Error 2", "Time 2", "MFLOPS 2", "Ratio T1/T2*100");


    for (std::size_t m=MIN_M, n=MIN_N;
         m<=MAX_M && n<=MAX_N;
         m+=INC_M, n+=INC_N)
    {
        double maxMN = std::max(m,n);
        double minMN = std::min(m,n);
        double flops = maxMN*minMN*minMN
                     - ((minMN*minMN*minMN) / 3.0)
                     - (minMN*minMN) / 2.0;
        flops /= 1000000.0;

        auto A0   = A.dim(m, n);
        auto tst1 = test_mkl_lu(A0);
        auto tst2 = test_lu(A0, lu);

        fmt::printf("%5d %5d "
                    "%15.2e %15.2f %15.2f "
                    "%15.2e %15.2f %15.2f %15.2lf\n",
                    m, n,
                    tst1.first, tst1.second, flops/tst1.second,
                    tst2.first, tst2.second, flops/tst2.second,
                    tst1.second/tst2.second*100);
    }
}
