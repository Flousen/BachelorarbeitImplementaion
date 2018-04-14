#ifndef HPC_MATVEC_SWAP_HPP
#define HPC_MATVEC_SWAP_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/gematrix.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <typename VectorX,
          typename VectorY,
          Require< Dense<VectorX>,
                   Dense<VectorY>,
                   SameElementType<VectorX,VectorY>> = true>
void
swap(VectorX &&x, VectorY &&y)
{
    assert(x.length()==y.length());

    for (std::size_t i=0; i<x.length(); ++i) {
        std::swap(x(i), y(i));
    }
}

template <typename MatrixA,
          template<typename> class VectorP,
          Require< Ge<MatrixA>,
                   Dense<VectorP<std::size_t>> > = true>
void
swap(const VectorP<std::size_t> &p, std::size_t i0, std::size_t i1,
     MatrixA &&A)
{
    std::ptrdiff_t inc = i0<i1 ? 1 : -1;

    i1 += inc;

    constexpr std::size_t bf = 32;

    std::size_t m = A.numRows();
    std::size_t n = A.numCols();

    for (std::size_t j=0; j<A.numCols(); j+=bf) {
        auto A_ = A.block(0,j).dim(m,std::min(bf, n-j));
        for (std::size_t i=i0; i!=i1; i+=inc) {
            if (i!=p(i)) {
                swap(A_.row(i, 0), A_.row(p(i), 0));
            }
        }
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_SWAP_HPP
