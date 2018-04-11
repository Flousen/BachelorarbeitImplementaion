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
          Require< Ge<MatrixA<T>> > = true>
std::pair<double, double>
test_qr(const MatrixA<T> &A0)
{
    GeMatrix<double>      A(A0.numRows(), A0.numCols());
    DenseVector<double>   tau( std::min( A0.numRows(), A0.numCols() ) );

    copy(A0, A);

    test::WallTime<double> timer;

    timer.tic();
    qr_blk(A, tau);
    double time = timer.toc();
    
    double err  = qr_error(A0, A, tau);

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

    fmt::printf("%5s %5s "
                "%10s %10s %10s\n",
                "M", "N",
                "Error", "Time", "MFLOPS");


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
        auto tst1 = test_qr(A0);

        fmt::printf("%5d %5d "
                    "%10.2e %10.2f %10.2f \n",
                    m, n,
                    tst1.first, tst1.second, flops/tst1.second);
    }
}