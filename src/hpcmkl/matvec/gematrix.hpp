#ifndef HPC_MATVEC_GEMATRIX_HPP
#define HPC_MATVEC_GEMATRIX_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/tools/buffer.hpp>
#include <hpc/matvec/storage/mixin.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <typename T>
class GeMatrix
    : public Mixin_MatrixDim,
      public Mixin_Conj<false>,
      public Mixin_Data<T, false>,
      public Mixin_View<GeMatrix<T>>,
      public Mixin_MatrixViewSelect<GeMatrix<T>>,
      public Mixin_MatrixElementAccess<GeMatrix<T>>
{
    GeMatrix()                                                  = delete;
    GeMatrix(const GeMatrix &rhs)                               = delete;
    GeMatrix& operator=(const GeMatrix &rhs)                    = delete;
    GeMatrix& operator=(GeMatrix &&rhs)                         = delete;

    public:
        using is_GeMatrix = std::true_type;

        GeMatrix(std::size_t numRows, std::size_t numCols,
                 Order order = Order::ColMajor)
            : Mixin_MatrixDim(numRows, numCols, order),
              Mixin_Conj<false>(false),
              Mixin_Data<T, false>(numRows*numCols)
        {
        }

        GeMatrix(GeMatrix &&rhs)                                = default;
        ~GeMatrix()                                             = default;

};

template <typename T>
class GeMatrixView
    : public Mixin_MatrixDim,
      public Mixin_Conj<false>,
      public Mixin_Data<T, true>,
      public Mixin_View<GeMatrixView<T>>,
      public Mixin_MatrixViewSelect<GeMatrixView<T>>,
      public Mixin_MatrixElementAccess<GeMatrixView<T>>
{
    GeMatrixView()                                              = delete;
    GeMatrixView(const GeMatrixView &rhs)                       = delete;
    GeMatrixView& operator=(const GeMatrixView &rhs)            = delete;
    GeMatrixView& operator=(GeMatrixView &&rhs)                 = delete;

    public:
        using is_GeMatrix = std::true_type;
        using VectorView  = DenseVectorView<T>;

        GeMatrixView(std::size_t numRows, std::size_t numCols,
                     bool conj, T *data,
                     std::ptrdiff_t incRow, std::ptrdiff_t incCol)
            : Mixin_MatrixDim(numRows, numCols, incRow, incCol),
              Mixin_Conj<false>(conj),
              Mixin_Data<T, true>(data)
        {
        }

        GeMatrixView(GeMatrixView &&rhs)                        = default;
        ~GeMatrixView()                                         = default;
};

template <typename T>
class GeMatrixConstView
    : public Mixin_MatrixDim,
      public Mixin_Conj<true>,
      public Mixin_Data<const T, true>,
      public Mixin_View<GeMatrixConstView<T>>,
      public Mixin_MatrixViewSelect<GeMatrixConstView<T>>,
      public Mixin_MatrixElementAccess<GeMatrixConstView<T>>
{
    GeMatrixConstView()                                         = delete;
    GeMatrixConstView(const GeMatrixConstView &rhs)             = delete;
    GeMatrixConstView& operator=(const GeMatrixConstView &rhs)  = delete;
    GeMatrixConstView& operator=(GeMatrixConstView &&rhs)       = delete;

    public:
        using is_GeMatrix = std::true_type;
        using VectorView  = DenseVectorConstView<T>;

        GeMatrixConstView(std::size_t numRows, std::size_t numCols,
                          bool conj, const T *data,
                          std::ptrdiff_t incRow, std::ptrdiff_t incCol)
            : Mixin_MatrixDim(numRows, numCols, incRow, incCol),
              Mixin_Conj<true>(conj),
              Mixin_Data<const T, true>(data)
        {
        }

        GeMatrixConstView(GeMatrixConstView &&rhs)              = default;
        ~GeMatrixConstView()                                    = default;

};

/// Functions to create a view
/// --------------------------
///
template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_create(const MatrixA<T> &A)
{
    return GeMatrixConstView<T>(A.numRows(), A.numCols(), A.conj(), A.data(),
                                A.incRow(), A.incCol());
}

// allowing a chain like A.constView().view()
template <typename T>
auto
view_create(GeMatrixConstView<T> &A)
{
    return GeMatrixConstView<T>(A.numRows(), A.numCols(), A.conj(), A.data(),
                                A.incRow(), A.incCol());
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_create(MatrixA<T> &A)
{
    return GeMatrixView<T>(A.numRows(), A.numCols(), A.conj(), A.data(),
                           A.incRow(), A.incCol());
}

/// Functions to select a block view
/// --------------------------------
///
template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, std::size_t i, std::size_t j)
{
    // in case i+numRows or j+numCols overflow
//    fmt::printf("i=%d, A.numRows()=%d\n", i, A.numRows());
//    fmt::printf("j=%d, A.numCols()=%d\n", j, A.numCols());
    assert(i<=A.numRows());
    assert(j<=A.numCols());

    return MatrixA<T>(A.numRows()-i, A.numCols()-j, A.conj(),
                      A.data()+i*A.incRow()+j*A.incCol(),
                      A.incRow(), A.incCol());
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, std::size_t i, std::size_t j,
            std::size_t numRows, std::size_t numCols)
{
    // in case i+numRows or j+numCols overflow
    assert(numRows<=A.numRows() && numCols<=A.numCols());
    assert(i+numRows<=A.numRows() && j+numCols<=A.numCols());

    return MatrixA<T>(numRows, numCols, A.conj(),
                      A.data()+i*A.incRow()+j*A.incCol(),
                      A.incRow(), A.incCol());
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, std::size_t i, std::size_t j,
            std::size_t numRows, std::size_t numCols,
            std::ptrdiff_t strideRow, std::ptrdiff_t strideCol)
{
    // in case i+numRows or j+numCols overflow
    assert(numRows<=A.numRows() && numCols<=A.numCols());
    assert(i+numRows<=A.numRows() && j+numCols<=A.numCols());

    if (strideRow<0) {
        i += numRows-1;
    }
    if (strideCol<0) {
        j += numCols-1;
    }

    numRows = (numRows+std::abs(strideRow)-1) / std::abs(strideRow);
    numCols = (numCols+std::abs(strideCol)-1) / std::abs(strideCol);


    return MatrixA<T>(numRows, numCols, A.conj(),
                      A.data()+i*A.incRow()+j*A.incCol(),
                      strideRow*A.incRow(), strideCol*A.incCol());
}

/// Functions to select transposed view or conjugated view
/// ------------------------------------------------------
///
template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, Trans)
{
    return MatrixA<T>(A.numCols(), A.numRows(), A.conj(), A.data(),
                      A.incCol(), A.incRow());
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, Conj)
{
    return MatrixA<T>(A.numRows(), A.numCols(), !A.conj(), A.data(),
                      A.incRow(), A.incCol());
}

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, ConjTrans)
{
    return MatrixA<T>(A.numCols(), A.numRows(), !A.conj(), A.data(),
                      A.incCol(), A.incRow());
}

/// Functions to select a vector view
/// ---------------------------------
/// - select a row
template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, Row, std::size_t i)
{
    assert(i<A.numRows());
    return SubVector<MatrixA<T>>(A.numCols(), A.conj(), A.data()+i*A.incRow(),
                                 A.incCol());
}

/// - select a column
template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
auto
view_select(MatrixA<T> &&A, Col, std::size_t j)
{
    assert(j<A.numCols());
    return SubVector<MatrixA<T>>(A.numRows(), A.conj(), A.data()+j*A.incCol(),
                                 A.incRow());
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_GEMATRIX_HPP
