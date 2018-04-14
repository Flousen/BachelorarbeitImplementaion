#ifndef HPC_MATVEC_STORAGE_MIXIN_HPP
#define HPC_MATVEC_STORAGE_MIXIN_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/views.hpp>
#include <hpc/tools/buffer.hpp>

namespace hpc { namespace matvec {

//------------------------------------------------------------------------------

struct Mixin_VectorDim
{
    const std::size_t       length_;
    const std::ptrdiff_t    inc_;

    Mixin_VectorDim(std::size_t length, std::ptrdiff_t inc)
        : length_(length), inc_(inc)
    {
        assert(inc_!=0);
    }

    std::size_t
    length() const
    {
        return length_;
    }

    std::ptrdiff_t
    inc() const
    {
        return inc_;
    }
};

//------------------------------------------------------------------------------

enum class Order { ColMajor, RowMajor };

struct Mixin_MatrixDim
{
    const std::size_t       numRows_, numCols_;
    const std::ptrdiff_t    incRow_, incCol_;

    Mixin_MatrixDim(std::size_t numRows, std::size_t numCols,
                   std::ptrdiff_t incRow, std::ptrdiff_t incCol)
        : numRows_(numRows), numCols_(numCols), incRow_(incRow), incCol_(incCol)
    {
    }

    Mixin_MatrixDim(std::size_t numRows, std::size_t numCols,
                    Order order)
        : numRows_(numRows), numCols_(numCols),
          incRow_(order==Order::ColMajor ? 1 : numCols),
          incCol_(order==Order::ColMajor ? numRows : 1)
    {
    }

    std::size_t
    numRows() const
    {
        return numRows_;
    }

    std::size_t
    numCols() const
    {
        return numCols_;
    }

    std::ptrdiff_t
    incRow() const
    {
        return incRow_;
    }

    std::ptrdiff_t
    incCol() const
    {
        return incCol_;
    }

};

//------------------------------------------------------------------------------

enum class UpLo { Lower, LowerUnit, Upper, UpperUnit };

struct Mixin_UpLo
{
    UpLo upLo;

    Mixin_UpLo(UpLo upLo)
        : upLo(upLo)
    {
    }

    bool
    is_lower() const
    {
        return upLo==UpLo::Lower || upLo==UpLo::LowerUnit;
    }

    bool
    is_upper() const
    {
        return !is_lower();
    }

    bool
    is_unit() const
    {
        return upLo==UpLo::LowerUnit || upLo==UpLo::UpperUnit;
    }

    bool
    is_nonUnit() const
    {
        return !is_unit();
    }
};

//------------------------------------------------------------------------------

template <bool hasConjVar>
struct Mixin_Conj
{
    const bool conj_;

    Mixin_Conj(bool conj)
        : conj_(conj)
    {
    }

    bool
    conj() const
    {
        return conj_;
    }
};

template <>
struct Mixin_Conj<false>
{
    Mixin_Conj(bool conj)
    {
        assert(!conj);
    }

    static constexpr bool
    conj()
    {
        return false;
    }
};

//------------------------------------------------------------------------------

template <typename T, bool is_view>
struct Mixin_Data
{
    T * const           data_;

    Mixin_Data(T *data)
        : data_(data)
    {
    }

    T *
    data() const
    {
        return data_;
    }
};

template <typename T>
struct Mixin_Data<T,false>
{
    tools::Buffer<T>    data_;

    Mixin_Data(std::size_t size)
        : data_(size)
    {
    }

    T *
    data() const
    {
        return &data_[0];
    }
};

//------------------------------------------------------------------------------

template <typename Derived>
struct Mixin_View
{
    Derived &
    This()
    {
        return *static_cast<Derived *>(this);
    }

    const Derived &
    This() const
    {
        return *static_cast<const Derived *>(this);
    }


    template <typename... Args>
    auto
    view(Args... args)
    {
        return hpc::matvec::view(This(), args...);
    }

    template <typename... Args>
    auto
    view(Args... args) const
    {
        return hpc::matvec::view(This(), args...);
    }

    template <typename... Args>
    auto
    constView(Args... args) const
    {
        return hpc::matvec::constView(This(), args...);
    }
};

//------------------------------------------------------------------------------

template <typename Derived>
struct Mixin_VectorElementAccess
{
    Derived &
    This()
    {
        return *static_cast<Derived *>(this);
    }

    const Derived &
    This() const
    {
        return *static_cast<const Derived *>(this);
    }

    auto &
    operator()(std::size_t i)
    {
        return This().data()[i*This().inc()];
    }

    const auto &
    operator()(std::size_t i) const
    {
        return This().data()[i*This().inc()];
    }

};

//------------------------------------------------------------------------------

template <typename Derived>
struct Mixin_MatrixElementAccess
{

    Derived &
    This()
    {
        return *static_cast<Derived *>(this);
    }

    const Derived &
    This() const
    {
        return *static_cast<const Derived *>(this);
    }

    auto &
    operator()(std::size_t i, std::size_t j)
    {
        return This().data()[i*This().incRow()+j*This().incCol()];
    }

    const auto &
    operator()(std::size_t i, std::size_t j) const
    {
        return This().data()[i*This().incRow()+j*This().incCol()];
    }
};

//------------------------------------------------------------------------------

template <typename Derived>
struct Mixin_VectorViewSelect
{
    Derived &
    This()
    {
        return *static_cast<Derived *>(this);
    }

    const Derived &
    This() const
    {
        return *static_cast<const Derived *>(this);
    }

    // block
    auto
    block(std::size_t i) const
    {
        return hpc::matvec::view(This(), i);
    }

    auto
    block(std::size_t i)
    {
        return hpc::matvec::view(This(), i);
    }

    // dim
    auto
    dim(std::size_t length) const
    {
        return hpc::matvec::view(This(), 0, length);
    }

    auto
    dim(std::size_t length)
    {
        return hpc::matvec::view(This(), 0, length);
    }
};

//------------------------------------------------------------------------------

template <typename Derived>
struct Mixin_MatrixViewSelect
{
    Derived &
    This()
    {
        return *static_cast<Derived *>(this);
    }

    const Derived &
    This() const
    {
        return *static_cast<const Derived *>(this);
    }

    // row view
    auto
    row(std::size_t i, std::size_t j)
    {
        return view(This().block(i,j), hpc::matvec::Row::view, 0);
    }

    auto
    row(std::size_t i, std::size_t j) const
    {
        return view(This().block(i,j), hpc::matvec::Row::view, 0);
    }

    // col view
    auto
    col(std::size_t i, std::size_t j) const
    {
        return view(This().block(i,j), hpc::matvec::Col::view, 0);
    }

    auto
    col(std::size_t i, std::size_t j)
    {
        return view(This().block(i,j), hpc::matvec::Col::view, 0);
    }

    // block
    auto
    block(std::size_t i, std::size_t j) const
    {
        return hpc::matvec::view(This(), i, j);
    }

    auto
    block(std::size_t i, std::size_t j)
    {
        return hpc::matvec::view(This(), i, j);
    }

    // dim
    auto
    dim(std::size_t numRows, std::size_t numCols) const
    {
        return hpc::matvec::view(This(), 0, 0, numRows, numCols);
    }

    auto
    dim(std::size_t numRows, std::size_t numCols)
    {
        return hpc::matvec::view(This(), 0, 0, numRows, numCols);
    }
};

} } // namespace matvec, hpc

#endif // HPC_MATVEC_STORAGE_MIXIN_HPP
