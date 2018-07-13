#include <complex>
#include <functional>
#include <random>

#include <printf.hpp>

#include <hpc/matvec.hpp>
#include <hpc/matvec/test/error.hpp>
#include <hpc/matvec/test/rand.hpp>
#include <hpc/matvec/test/walltime.hpp>

#include <hpc/mklblas.hpp>

namespace hpc { namespace matvec {

template <typename T, template<typename> class MatrixA,
          typename qr_func,
          Require< Ge<MatrixA<T>> > = true>
std::pair<double, double>
test_qr(const MatrixA<T> &A0, qr_func qr)
{
    GeMatrix<double>  A(A0.numRows(), A0.numCols(),Order::ColMajor);
    DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));

    copy(A0, A);

    test::WallTime<double> timer;
    //fmt::printf("befor\n"); print(A);

    timer.tic();
    qr(A, tauA);
    double time = timer.toc();

    //fmt::printf("after\n"); print(A);

    double err  = qr_error(A0, A, tauA);


    return std::pair<double, double>(err, time);
}

} } // namespace matvec, hpc

#define MIN_M 10
#define MIN_N 10
#define INC_M 20
#define INC_N 10 
#define MAX_M 1000
#define MAX_N 1000

int
main()
{
  using namespace hpc::matvec;

  GeMatrix<double> A(MAX_M, MAX_N, Order::ColMajor);
  DenseVector<double> tauA(std::min(A.numRows(),A.numCols()));
  
  test::rand(A);

  auto qr_unblk = hpc::mklblas::qr_unblk<GeMatrix<double> &, DenseVector<double> &>;
  auto qr_unblk_ref = hpc::mklblas::qr_unblk_ref<GeMatrix<double> &, DenseVector<double> &>;

  fmt::printf("%5s %5s "
              "%10s %10s %10s "
              "%10s %10s %10s\n",
              "M", "N",
              "Error unblk", "Time ", "MFLOPS ",
              "Error ref", "Time ", "MFLOPS ");


  for (std::size_t m=MIN_M, n=MIN_N;
       m<=MAX_M && n<=MAX_N;
       m+=INC_M, n+=INC_N)
  {
    double maxMN = std::max(m,n);
    double minMN = std::min(m,n);
    double flops = n*(23/6 + m + n/2 + n*(m-n/3) + n*(5/6 + n/2 + m - n/3));
    //double flops = n*n* ( m -  n / 3 );
    //double flops = maxMN*minMN*minMN
    //             - ((minMN*minMN*minMN) / 3.0)
    //             - (minMN*minMN) / 2.0;
    //flops *= 2;
    flops /= 1000000.0;

    auto A0   = A.dim(m, n);
    fmt::printf("%5d %5d ", m, n);
    auto tst1 = test_qr(A0, qr_unblk);
    auto tst2 = test_qr(A0, qr_unblk_ref);
    std::fprintf(stderr, "%s %d\n", "n = ",n);
    
    fmt::printf( "%lf %10.2f %10.2f ",
                tst1.first, tst1.second, flops/tst1.second);
    
    fmt::printf( "%lf %10.2f %10.2f",
                tst2.first, tst2.second, flops/tst2.second);

    fmt::printf("  25600");
    fmt::printf("\n");
  }
  std::fprintf(stderr, "%s", "successful\n");
}
