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
          Require< Ge<MatrixA<T> >,
                   Ge<MatrixB<T> >,
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

template <typename Alpha,
          typename T, template<typename> class MatrixA,
                      typename MatrixB,
          Require< Tr<MatrixA<T>>,
                   Ge<MatrixB>
                 > = true>
void
mm(const Alpha &alpha, const MatrixA<T> &A, MatrixB &&B, bool trans = false)
{
    assert(A.numCols()==A.numRows());
    assert(A.numCols()==B.numRows());

    GeMatrix<T> TrTmp(A.numRows(),A.numCols());
    copy(A,TrTmp);

    GeMatrix<T> BTmp(B.numRows(),B.numCols());
    copy(B,BTmp);

    if ( trans ){
      ulmblas::gemm(B.numRows(), B.numCols(), A.numCols(),
                    alpha,
                    TrTmp.conj(), TrTmp.data(), TrTmp.incCol(), TrTmp.incRow(),
                    B.conj(), B.data(), B.incRow(), B.incCol(),
                    T(0),
                    BTmp.data(), BTmp.incRow(), BTmp.incCol());
    } else{
      ulmblas::gemm(B.numRows(), B.numCols(), A.numCols(),
                    alpha,
                    TrTmp.conj(), TrTmp.data(), TrTmp.incRow(), TrTmp.incCol(),
                    B.conj(), B.data(), B.incRow(), B.incCol(),
                    T(0),
                    BTmp.data(), BTmp.incRow(), BTmp.incCol());
    }
    copy(BTmp, B);
}

template <typename Alpha,
          typename T, typename MatrixA,
                      template<typename> class MatrixB,
          Require< Ge<MatrixA>,
                   Tr<MatrixB<T>>
                 > = true>
void
mm(const Alpha &alpha, MatrixA &&A, const MatrixB<T> &B, bool trans = false)
{
    assert(A.numCols()==B.numRows());
    assert(B.numCols()==B.numRows());

    GeMatrix<T> TrTmp(B.numRows(),B.numCols());
    copy(B,TrTmp);

    GeMatrix<T> ATmp(A.numRows(),A.numCols());
   
    if ( trans ){
      ulmblas::gemm(A.numRows(), A.numCols(), B.numCols(),
                  alpha,
                  A.conj(), A.data(), A.incRow(), A.incCol(),
                  TrTmp.conj(), TrTmp.data(), TrTmp.incCol(), TrTmp.incRow(),
                  T(0),
                  ATmp.data(), ATmp.incRow(), ATmp.incCol());
    } else{
      ulmblas::gemm(A.numRows(), A.numCols(), B.numCols(),
                  alpha,
                  A.conj(), A.data(), A.incRow(), A.incCol(),
                  TrTmp.conj(), TrTmp.data(), TrTmp.incRow(), TrTmp.incCol(),
                  T(0),
                  ATmp.data(), ATmp.incRow(), ATmp.incCol());
    }

    copy(ATmp, A);
    //mm(alpha, B.view(Trans::view), A.view(Trans::view));
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_MM_HPP
