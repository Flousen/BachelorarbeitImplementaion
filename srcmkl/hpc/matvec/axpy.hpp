#ifndef HPC_MATVEC_AXPY_HPP
#define HPC_MATVEC_AXPY_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/axpy.hpp>

namespace hpc { namespace matvec {

template <typename Alpha,
          typename T, template<typename> class VectorX,
                      typename VectorY,
          Require< Dense<VectorX<T>>,
                   Dense<VectorY> > = true>
void
axpy(const Alpha &alpha, const VectorX<T> &x, VectorY &&y)
{
    assert(x.length()==y.length());

    ulmblas::axpy(x.length(), alpha, x.conj(), x.data(), x.inc(),
                  y.data(), y.inc());
}

template <typename Alpha,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixB<T>> > = true>
void
axpy(const Alpha &alpha, const MatrixA<T> &A, MatrixB<T> &B)
{
    assert(A.numRows()==B.numRows() && A.numCols()==A.numCols());

    ulmblas::geaxpy(A.numRows(), A.numCols(), alpha,
                    A.conj(), A.data(), A.incRow(), A.incCol(),
                    B.data(), B.incRow(), B.incCol());
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_AXPY_HPP
