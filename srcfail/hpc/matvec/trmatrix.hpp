#ifndef HPC_MATVEC_TRMATRIX_HPP
#define HPC_MATVEC_TRMATRIX_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/gematrix.hpp>

namespace hpc { namespace matvec {

template <typename T>
class TrMatrixView
    : public Mixin_UpLo,
      public GeMatrixView<T>
{
    TrMatrixView()                                              = delete;
    TrMatrixView(const TrMatrixView &rhs)                       = delete;
    TrMatrixView& operator=(const TrMatrixView &rhs)            = delete;
    TrMatrixView& operator=(TrMatrixView &&rhs)                 = delete;

    public:
        using is_GeMatrix = std::false_type;
        using is_TrMatrix = std::true_type;

        TrMatrixView(std::size_t numRows, std::size_t numCols,
                     UpLo upLo, bool conj, T *data,
                     std::ptrdiff_t incRow, std::ptrdiff_t incCol)
            : Mixin_UpLo(upLo),
              GeMatrixView<T>(numRows, numCols, conj, data, incRow, incCol)
        {
        }

        TrMatrixView(TrMatrixView &&rhs)                        = default;
        ~TrMatrixView()                                         = default;
};

template <typename T>
class TrMatrixConstView
    : public Mixin_UpLo,
      public GeMatrixConstView<T>
{
    TrMatrixConstView()                                         = delete;
    TrMatrixConstView(const TrMatrixConstView &rhs)             = delete;
    TrMatrixConstView& operator=(const TrMatrixConstView &rhs)  = delete;
    TrMatrixConstView& operator=(TrMatrixConstView &&rhs)       = delete;

    public:
        using is_GeMatrix = std::false_type;
        using is_TrMatrix = std::true_type;

        TrMatrixConstView(std::size_t numRows, std::size_t numCols,
                          bool conj, const T *data,
                          std::ptrdiff_t incRow, std::ptrdiff_t incCol)
            : Mixin_UpLo(upLo),
              GeMatrixConstView<T>(numRows, numCols, conj, data, incRow, incCol)
        {
        }

        TrMatrixConstView(TrMatrixConstView &&rhs)              = default;
        ~TrMatrixConstView()                                    = default;

};

/// TrMatrixView and TrMatrixConstView can only be created from corresponding
/// GeMatrix types

template <typename T>
auto
view_select(GeMatrixView<T> &&A, UpLo upLo)
{
    return TrMatrixView<T>(A.numRows(), A.numCols(), upLo, A.conj(), A.data(),
                           A.incRow(), A.incCol());
}

template <typename T>
auto
view_select(GeMatrixConstView<T> &&A, UpLo upLo)
{
    return TrMatrixConstView<T>(A.numRows(), A.numCols(), upLo, A.conj(),
                                A.data(), A.incRow(), A.incCol());
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_GEMATRIX_HPP
