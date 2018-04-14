#ifndef HPC_MATVEC_SCAL_HPP
#define HPC_MATVEC_SCAL_HPP

#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/scal.hpp>

namespace hpc { namespace matvec {

template <typename Alpha, typename VectorX,
          Require< Dense<VectorX> > = true>
void
scal(const Alpha &alpha, VectorX &&x)
{
    ulmblas::scal(x.length(), alpha, x.data(), x.inc());
}

/*
template <typename Alpha, typename T, template<typename> class VectorX,
          Require< Dense<VectorX<T>> > = true>
void
scal(const Alpha &alpha, VectorX<T> &x)
{
    ulmblas::scal(x.length(), alpha, x.data(), x.inc());
}
*/

template <typename Alpha, typename T, template<typename> class MatrixA,
          Require< Ge<MatrixA<T>> > = true>
void
scal(const Alpha &alpha, MatrixA<T> &A)
{
    ulmblas::gescal(A.numRows(), A.numCols(), alpha,
                    A.data(), A.incRow(), A.incCol());
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_SCAL_HPP
