#ifndef HPC_MATVEC_NORM_HPP
#define HPC_MATVEC_NORM_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

namespace hpc { namespace matvec {

/// Functions `hpc::matvec::mv`
/// ---------------------------
///
/// - gemv operation $y \leftarrow \beta y + \alpha A x$
///
template < typename T, template<typename> class MatrixA,
          Require< Ge<MatrixA<T>> > = true>
T
norm_inf(const MatrixA<T> &A)
{
  T res = 0;
  for (size_t i=0; i<A.numRows(); ++i) {
    T asum = 0;
    for (size_t j=0; j<A.numCols(); ++j) {
      asum += fabs(A(i,j));
    }
    if (asum>res) {
      res = asum;
    }
  }
  return res;
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_NORM_HPP
