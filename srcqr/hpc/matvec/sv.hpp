#ifndef HPC_MATVEC_SV_HPP
#define HPC_MATVEC_SV_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/axpy.hpp>
#include <hpc/matvec/dot.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

/// - trsv operation $b \leftarrow A^{-1} b$
template <typename T, template<typename> class MatrixA,
                      typename                 VectorB,
          Require< Tr<MatrixA<T>>,
                   Dense<VectorB>,
                   SameElementType<MatrixA<T>, VectorB>
                 > = true>
void
sv(const MatrixA<T> &A, VectorB &&b)
{
    assert(A.numRows()==A.numCols());
    assert(A.numRows()==b.length());

    if (A.is_lower()) {
        if (A.incRow()<A.incCol()) {
            for (std::size_t j=0; j<A.numCols(); ++j) {
                if (A.is_nonUnit()) {
                    b(j) /= A(j,j);
                }
                if (j+1<A.numRows()) {
                    axpy(-b(j), A.col(j+1,j), b.block(j+1));
                }
            }
        } else {
            for (std::size_t i=0; i<A.numRows(); ++i) {
                b(i) -= dot(A.row(i,0).dim(i), b.dim(i));
                if (A.is_nonUnit()) {
                    b(i) /= A(i,i);
                }
            }
        }
    } else {
        assert(0); // not implemented
    }
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_SV_HPP
