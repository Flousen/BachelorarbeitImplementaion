#ifndef HPC_MATVEC_APPLY_HPP
#define HPC_MATVEC_APPLY_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <template<typename> class VectorX, typename T, typename Func,
          Require< Dense<VectorX<T>> > = true>
void
apply(const VectorX<T> &x, Func func)
{
    for (std::size_t i=0; i<x.length(); ++i) {
        func(i);
    }
}

template <template<typename> class MatrixA, typename T, typename Func,
          Require< Ge<MatrixA<T>> > = true>
void
apply(const MatrixA<T> &A, Func func)
{
    if (A.incRow()<A.incCol()) {
        for (std::size_t j=0; j<A.numCols(); ++j) {
            for (std::size_t i=0; i<A.numRows(); ++i) {
                func(i,j);
            }
        }
    } else {
        for (std::size_t i=0; i<A.numRows(); ++i) {
            for (std::size_t j=0; j<A.numCols(); ++j) {
                func(i,j);
            }
        }
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_APPLY_HPP
