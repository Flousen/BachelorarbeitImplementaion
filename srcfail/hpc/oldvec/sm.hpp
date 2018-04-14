#ifndef HPC_MATVEC_SM_HPP
#define HPC_MATVEC_SM_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/sv.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {


/// - trsm operation $B \leftarrow \alpha A^{-1} B$
template <typename Alpha,
          typename T, template<typename> class MatrixA,
                      typename                 MatrixB,
          Require< Tr<MatrixA<T>>,
                   Ge<MatrixB>,
                   SameElementType<MatrixA<T>, MatrixB>
                 > = true>
void
sm(const Alpha &alpha, const MatrixA<T> &A, MatrixB &&B)
{
    assert(A.numRows()==A.numCols());
    assert(A.numRows()==B.numRows());

    scal(alpha, B);
    if (alpha==Alpha(0)) {
        return;
    }
    for (std::size_t j=0; j<B.numCols(); ++j) {
        sv(A, B.col(0,j));
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_SM_HPP
