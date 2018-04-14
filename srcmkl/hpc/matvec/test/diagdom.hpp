#ifndef HPC_ULMBLAS_TEST_DIAGDOM_HPP
#define HPC_ULMBLAS_TEST_DIAGDOM_HPP

#include <cfloat>
#include <cmath>
#include <complex>
#include <hpc/matvec.hpp>

namespace hpc { namespace matvec { namespace test {

template <typename MatrixA,
          Require< Ge<MatrixA> > = true>
void
diagdom(MatrixA &&A)
{
    assert(A.numRows()==A.numCols());
    using T = ElementType<MatrixA>;

    for (std::size_t i=0; i<A.numRows(); ++i) {
        T aSum = T(0);
        for (std::size_t j=0; j<A.numCols(); ++j) {
            if (i!=j) {
                aSum += std::abs(A(i,j));
            }
        }
        A(i,i) = aSum + T(1);
    }
}


template <typename MatrixA,
          Require< Tr<MatrixA> > = true>
void
diagdom(MatrixA &&A)
{
    assert(A.numRows()==A.numCols());
    using T = ElementType<MatrixA>;

    if (A.is_lower()) {
        for (std::size_t i=0; i<A.numRows(); ++i) {
            T aSum = T(0);
            for (std::size_t j=0; j<i; ++j) {
                aSum += std::abs(A(i,j));
            }
            if (A.is_nonUnit()) {
                A(i,i) = aSum;
            } else {
                for (std::size_t j=0; j<i; ++j) {
                    A(i,j) /= aSum;
                }
            }
        }
    } else {
        for (std::size_t i=0; i<A.numRows(); ++i) {
            T aSum = T(0);
            for (std::size_t j=i+1; j<A.numCols(); ++j) {
                aSum += std::abs(A(i,j));
            }
            if (A.is_nonUnit()) {
                A(i,i) = aSum;
            } else {
                for (std::size_t j=i+1; j<A.numCols(); ++j) {
                    A(i,j) /= aSum;
                }
            }
        }
    }
}

} } } // namespace test, matvec, hpc

#endif // HPC_ULMBLAS_TEST_DIAGDOM_HPP
