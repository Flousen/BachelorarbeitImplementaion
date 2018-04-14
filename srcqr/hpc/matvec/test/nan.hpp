#ifndef HPC_MATVEC_TEST_NAN_HPP
#define HPC_MATVEC_TEST_NAN_HPP

#include <cfloat>
#include <complex>

namespace hpc { namespace matvec { namespace test {

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
void
nan(MatrixA<T> &A)
{
    auto f = [](...) -> T
    {
        return std::nan("");
    };
    apply(A, f);
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<std::complex<T>>> > = true>
void
nan(MatrixA<std::complex<T>> &A)
{
    auto f = [](...) -> std::complex<T>
    {
        return std::complex<T>(std::nan(""),std::nan(""));
    };
    apply(A, f);
}


} } } // namespace test, matvec, hpc

#endif // HPC_MATVEC_TEST_NAN_HPP
