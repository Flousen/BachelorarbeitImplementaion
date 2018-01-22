#ifndef HPC_MATVEC_TEST_REF_HPP
#define HPC_MATVEC_TEST_REF_HPP

#include <complex>
#include <random>

namespace hpc { namespace matvec { namespace test {

// *Internal* reference implementation for gemm (if nothing better is available)
template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
                      typename MatrixC,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixB<T>>,
                   Ge<MatrixC>
                 > = true>
void
mm_ref(const Alpha &alpha, const MatrixA<T> &A, const MatrixB<T> &B,
       const Beta &beta, MatrixC &&C)
{
    assert(A.numCols()==B.numRows());
    assert(C.numRows()==A.numRows());
    assert(C.numCols()==B.numCols());

    std::size_t m = C.numRows();
    std::size_t n = C.numCols();
    std::size_t k = A.numCols();

    scal(beta, C);
    if (k==0 || alpha==Alpha(0)) {
        return;
    }
    for (size_t j=0; j<n; ++j) {
        for (size_t l=0; l<k; ++l) {
            for (size_t i=0; i<m; ++i) {
                C(i,j) += alpha*A(i,l)*B(l,j);
            }
        }
    }
}

} } } // namespace test, matvec, hpc

#endif // HPC_MATVEC_TEST_REF_HPP
