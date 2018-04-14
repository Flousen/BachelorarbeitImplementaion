#ifndef HPC_MATVEC_COPY_HPP
#define HPC_MATVEC_COPY_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/copy.hpp>

namespace hpc { namespace matvec {

template <typename T, template<typename> class VectorX,
                      typename VectorY,
          Require< Dense<VectorX<T>>,
                   Dense<VectorY> > = true>
void
copy(const VectorX<T> &x, VectorY &&y)
{
    assert(x.length()==y.length());
    ulmblas::copy(x.length(), x.conj(), x.data(), x.inc(), y.data(), y.inc());
}

template <typename T,
          template<typename> class MatrixA, template<typename> class MatrixB,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixB<T>> > = true>
void
copy(const MatrixA<T> &A, MatrixB<T> &B)
{
    assert(A.numRows()==B.numRows() && A.numCols()==A.numCols());
    ulmblas::gecopy(A.numRows(), A.numCols(),
                    A.conj(), A.data(), A.incRow(), A.incCol(),
                    B.data(), B.incRow(), B.incCol());
}

template <typename T,
          template<typename> class MatrixA, template<typename> class MatrixB,
          Require< Tr<MatrixA<T>>,
                   Ge<MatrixB<T>> > = true>
void
copy(const MatrixA<T> &A, MatrixB<T> &B)
{
    assert(A.numRows()==B.numRows() && A.numCols()==A.numCols());

    for (std::size_t i=0; i<A.numRows(); ++i) {
      for (std::size_t j=0; j<A.numCols(); ++j) {
        if (j>i) {
          B(i,j) = A.is_upper() ? A(i,j) : T(0);
        } else if (i==j) {
          B(i,j) = A.is_nonUnit() ? A(i,j) : T(1);
        } else {
          B(i,j) = A.is_lower() ? A(i,j) : T(0);
        }
      }
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_COPY_HPP
