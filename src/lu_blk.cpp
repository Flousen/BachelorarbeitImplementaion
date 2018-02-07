#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>

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

} } // namespace matvec, hpc

#define MIN_M 10
#define MIN_N 10
#define INC_M 10
#define INC_N 10
#define MAX_M 1000
#define MAX_N 1000

int
main()
{
    using namespace hpc::matvec;

    GeMatrix<double> A(MAX_M, MAX_N);

    test::rand(A);

    auto lu1 = lu_blk_var1<GeMatrix<double> &, DenseVector<std::size_t> &>;

    // For now we just call the same variant twice.  For quiz06 simply call
    // function lu_blk_var2 instead of lu_blk_var1:
    auto lu2 = lu_blk_var1<GeMatrix<double> &, DenseVector<std::size_t> &>;
    //auto lu2 = lu_blk_var2<GeMatrix<double> &, DenseVector<std::size_t> &>;

    fmt::printf("%5s %5s "
                "%10s %10s %10s "
                "%10s %10s %10s\n",
                "M", "N",
                "Error 1", "Time 1", "MFLOPS 1",
                "Error 2", "Time 2", "MFLOPS 2");


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
        auto tst1 = test_lu(A0, lu1);
        auto tst2 = test_lu(A0, lu2);

        fmt::printf("%5d %5d "
                    "%10.2e %10.2f %10.2f "
                    "%10.2e %10.2f %10.2f\n",
                    m, n,
                    tst1.first, tst1.second, flops/tst1.second,
                    tst2.first, tst2.second, flops/tst2.second);
    }
}
