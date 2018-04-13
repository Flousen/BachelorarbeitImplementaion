#ifndef HPC_MATVEC_RANK1_HPP
#define HPC_MATVEC_RANK1_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <typename Alpha,
          typename VectorX,
          typename VectorY,
          typename MatrixA,
          Require< Dense<VectorX>,
                   Dense<VectorY>,
                   Ge<MatrixA>
          > = true>
void
rank1(const Alpha &alpha, const VectorX &x, const VectorY &y,
      MatrixA &&A)
{
    if (A.incRow()<A.incCol()) {
        for (std::size_t j=0; j<A.numCols(); ++j) {
            axpy(alpha*y(j), x, A.col(0,j));
        }
    } else {
        for (std::size_t i=0; i<A.numRows(); ++i) {
            axpy(alpha*x(i), y, A.row(i,0));
        }
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_RANK1_HPP
