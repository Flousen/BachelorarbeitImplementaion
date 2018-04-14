#ifndef HPC_MATVEC_TEST_NORMINF_HPP
#define HPC_MATVEC_TEST_NORMINF_HPP

#include <cfloat>
#include <cmath>
#include <complex>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec { namespace test {

template <typename T, template<typename> class MatrixA,
          Require< Ge<MatrixA<T>> > = true>
auto
norminf(const MatrixA<T> &A)
{
    Real<T> res = 0;
    for (size_t i=0; i<A.numRows(); ++i) {
        Real<T> asum = 0;
        for (size_t j=0; j<A.numCols(); ++j) {
            asum += std::abs(A(i,j));
        }
        if (std::isnan(asum)) {
            return asum;
        }
        if (asum>res) {
            res = asum;
        }
    }
    return res;
}


} } } // namespace test, matvec, hpc

#endif // HPC_MATVEC_TEST_NORMINF_HPP
