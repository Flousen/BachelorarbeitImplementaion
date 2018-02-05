#ifndef HPC_MATVEC_TRMV_HPP
#define HPC_MATVEC_TRMV_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/fused.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/gemm.hpp>

namespace hpc { namespace matvec {

/// Functions `hpc::matvec::mv`
/// ---------------------------
///
/// - gemv operation $y \leftarrow \beta y + \alpha A x$
///
template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class VectorX,
                      typename                 VectorY,
          Require< Ge<MatrixA<T>>,
                   Dense<VectorX<T>>,
                   Dense<VectorY>,
                   SameElementType<VectorX<T>, VectorY>
                 > = true>
void
trmv(const Alpha &alpha, const MatrixA<T> &A, const VectorX<T> &x,
   const Beta &beta, VectorY &&y)
{
    assert(A.numRows()==y.length());
    assert(A.numCols()==x.length());
    


    scal(beta, y);
    if (A.incRow()<A.incCol()) {
        // col major
        constexpr std::size_t  bf = 4;
        std::size_t            nb = A.numCols()/bf;
        for (std::size_t j=0; j<nb; ++j) {
            fused<bf>::axpy(alpha, A.block(0,j*bf).dim(A.numRows(),bf),
                            x.block(j*bf).dim(bf), y);
        }
        for (std::size_t j=nb*bf; j<A.numCols(); ++j) {
            axpy(alpha*x(j), A.col(0,j), y);
        }
    } else {
        // row major
        constexpr std::size_t  bf = 4;
        std::size_t            mb = A.numRows()/bf;
        for (std::size_t i=0; i<mb; ++i) {
            fused<bf>::dot(alpha, A.block(i*bf,0).dim(bf, A.numCols()),
                           x, y.block(i*bf).dim(bf));
        }
         for (std::size_t i=mb*bf; i<A.numRows(); ++i) {
            y(i) += alpha*dot(A.row(i,0),x);
        }
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_TRMV_HPP
