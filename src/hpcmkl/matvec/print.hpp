#ifndef HPC_MATVEC_PRINT_HPP
#define HPC_MATVEC_PRINT_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/gematrix.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/print.hpp>

namespace hpc { namespace matvec {

template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
void
print(const VectorX<T> &x, const char *format)
{
    ulmblas::print(x.length(), x.data(), x.inc(), format);
}

template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
void
print(const VectorX<T> &x)
{
    ulmblas::print(x.length(), x.conj(), x.data(), x.inc());
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
void
print(const MatrixA<T> &A, const char *format)
{
    ulmblas::geprint(A.numRows(), A.numCols(),
                     A.conj(), A.data(), A.incRow(), A.incCol(),
                     format);
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
void
print(const MatrixA<T> &A)
{
    ulmblas::geprint(A.numRows(), A.numCols(),
                     A.conj(), A.data(), A.incRow(), A.incCol());
}

template <template<typename> class MatrixA, typename T,
          Require< Tr<MatrixA<T>> > = true>
void
print(const MatrixA<T> &A, const char *format = " %8.2lf")
{
    for (std::size_t i=0; i<A.numRows(); ++i) {
        for (std::size_t j=0; j<A.numCols(); ++j) {
            if (j>i) {
                fmt::printf(format, A.is_upper() ? A(i,j) : T(0));
            } else if (i==j) {
                fmt::printf(format, A.is_nonUnit() ? A(i,j) : T(1));
            } else {
                fmt::printf(format, A.is_lower() ? A(i,j) : T(0));
            }
        }
        fmt::printf("\n");
    }
    fmt::printf("\n");
    printf("Actual storage:\n");
    ulmblas::geprint(A.numRows(), A.numCols(),
                     A.conj(), A.data(), A.incRow(), A.incCol());
}



} } // namespace matvec, hpc

#endif // HPC_MATVEC_PRINT_HPP
