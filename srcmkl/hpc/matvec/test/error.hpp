#ifndef HPC_MATVEC_TEST_ERROR_HPP
#define HPC_MATVEC_TEST_ERROR_HPP

#include <cmath>
#include <limits>
#include <hpc/matvec.hpp>
#include <hpc/matvec/test/norminf.hpp>

namespace hpc { namespace matvec { namespace test {

namespace error_estimate {

template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
                      template<typename> class MatrixC0,
                      template<typename> class MatrixRefC,
                      template<typename> class MatrixTstC,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixB<T>>,
                   Ge<MatrixC0<T>>,
                   Ge<MatrixRefC<T>>,
                   Ge<MatrixTstC<T>> > = true>
auto
gemm(const Alpha  &alpha,
     const MatrixA<T> &A, const MatrixB<T> &B, const MatrixC0<T> &C0,
     const Beta &beta,
     const MatrixRefC<T> &RefC, MatrixTstC<T> &TstC)
{
    axpy(T(-1), RefC, TstC);

    Real<T>     normD = norminf(TstC);
    std::size_t N     = std::max(TstC.numRows(),
                                 std::max(TstC.numCols(), A.numCols()));

    if (isnan(normD)) {
        return normD;
    }

    if (normD==Real<T>(0)) {
        return Real<T>(0);
    }

    Real<T> normA = 0;
    Real<T> normB = 0;

    if (alpha!=Alpha(0)) {
        normB   = norminf(B);
        normA   = norminf(A);
        normA  *= std::max(std::abs(Alpha(1)), std::abs(alpha));
        if (normA==Real<T>(0)) {
            normA = Real<T>(1);
        }
    }

    Real<T> normC0 = 0;
    if (beta!=Beta(0)) {
        normC0 = norminf(C0);
        normC0 *= std::max(std::abs(Beta(1)), std::abs(beta));
        if (normC0==Real<T>(0)) {
            normC0 = Real<T>(1);
        }
    }

    normA  = std::max(Real<T>(1), normA);
    normB  = std::max(Real<T>(1), normB);
    normC0 = std::max(Real<T>(1), normC0);

    auto eps = std::numeric_limits<Real<T>>::epsilon();
    return normD/((normA*normB+normC0)*eps*N);
}

template <typename T, template<typename> class MatrixA,
                      template<typename> class MatrixLU,
                      template<typename> class VectorPiv,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixLU<T>>,
                   Dense<VectorPiv<std::size_t>> > = true>
auto
getrf(const MatrixA<T> &A0, const MatrixLU<T> &LU,
      const VectorPiv<std::size_t> &piv)
{
    std::size_t m = LU.numRows();
    std::size_t n = LU.numCols();
    std::size_t k = std::min(m,n);

    GeMatrix<T>     L(m, k), U(k, n), A(m,n);

    // copy L-part from A
    for (std::size_t l=0; l<k; ++l) {
        for (std::size_t i=0; i<m; ++i) {
            L(i,l) = (i>l)  ? LU(i,l) :
                     (i==l) ? T(1)   :
                              T(0);
        }
    }
    // copy U-part from A
    for (std::size_t j=0; j<n; ++j) {
        for (std::size_t l=0; l<k; ++l) {
            U(l,j) = (l>j)  ? T(0)
                            : LU(l,j);
        }
    }
    // A = L*U
    matvec::mm(T(1), L, U, T(0), A);
    // A = P^{-1}*A
    swap(piv, std::min(m,n)-1, 0, A);

    matvec::axpy(T(-1), A0, A);

    auto eps = std::numeric_limits<Real<T>>::epsilon();
    return norminf(A)/(norminf(A0)*eps*std::min(m,n));
}

} // namespace error_estimate

} } } // namespace test, matvec, hpc

#endif // HPC_MATVEC_TEST_ERROR_HPP
