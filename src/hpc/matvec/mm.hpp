#ifndef HPC_MATVEC_MM_HPP
#define HPC_MATVEC_MM_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/gemm.hpp>

#include <hpc/matvec/gematrix.hpp>
#include <hpc/matvec/copy.hpp>

namespace hpc { namespace matvec {

/// Functions `hpc::matvec::mm`
/// ---------------------------
///
/// - gemm operation $C \leftarrow \beta C + \alpha A B$
///
template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
                      typename MatrixC,
          Require< Ge<MatrixA<T>>,
                   Ge<MatrixB<T>>,
                   Ge<MatrixC>
                 > = true>
void
mm(const Alpha &alpha, const MatrixA<T> &A, const MatrixB<T> &B,
   const Beta &beta, MatrixC &&C)
{
    assert(A.numCols()==B.numRows());
    assert(C.numRows()==A.numRows());
    assert(C.numCols()==B.numCols());

    ulmblas::gemm(C.numRows(), C.numCols(), A.numCols(),
                  alpha,
                  A.conj(), A.data(), A.incRow(), A.incCol(),
                  B.conj(), B.data(), B.incRow(), B.incCol(),
                  beta,
                  C.data(), C.incRow(), C.incCol());
}

template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
                      typename MatrixC,
          Require< Tr<MatrixA<T>>,
                   Ge<MatrixB<T>>,
                   Ge<MatrixC>
                 > = true>
void
mm(const Alpha &alpha, const MatrixA<T> &A, const MatrixB<T> &B,
   const Beta &beta, MatrixC &&C)
{
    assert(A.numCols()==B.numRows());
    assert(C.numRows()==A.numRows());
    assert(C.numCols()==B.numCols());


    GeMatrix<T> At(A.numRows(),A.numCols());
    copy(A,At);

    ulmblas::gemm(C.numRows(), C.numCols(), At.numCols(),
                  alpha,
                  At.conj(), At.data(), At.incRow(), At.incCol(),
                  B.conj(), B.data(), B.incRow(), B.incCol(),
                  beta,
                  C.data(), C.incRow(), C.incCol());
}

template <typename Alpha, typename Beta,
          typename T, template<typename> class MatrixA,
                      template<typename> class MatrixB,
                      typename MatrixC,
          Require< Ge<MatrixA<T>>,
                   Tr<MatrixB<T>>,
                   Ge<MatrixC>
                 > = true>
void
mm(const Alpha &alpha, const MatrixA<T> &A, const MatrixB<T> &B,
   const Beta &beta, MatrixC &&C)
{
    assert(A.numCols()==B.numRows());
    assert(C.numRows()==A.numRows());
    assert(C.numCols()==B.numCols());

    GeMatrix<T> Bt(B.numRows(),B.numCols());
    copy(B,Bt);

    ulmblas::gemm(C.numRows(), C.numCols(), A.numCols(),
                  alpha,
                  A.conj(), A.data(), A.incRow(), A.incCol(),
                  Bt.conj(), Bt.data(), Bt.incRow(), Bt.incCol(),
                  beta,
                  C.data(), C.incRow(), C.incCol());
}


} } // namespace matvec, hpc

#endif // HPC_MATVEC_MM_HPP
