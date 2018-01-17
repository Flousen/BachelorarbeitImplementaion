#ifndef HPC_MATVEC_DENSEVECTOR_HPP
#define HPC_MATVEC_DENSEVECTOR_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/tools/buffer.hpp>
#include <hpc/matvec/storage/mixin.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <typename T>
class DenseVector
    : public Mixin_VectorDim,
      public Mixin_Conj<false>,
      public Mixin_Data<T, false>,
      public Mixin_View<DenseVector<T>>,
      public Mixin_VectorViewSelect<DenseVector<T>>,
      public Mixin_VectorElementAccess<DenseVector<T>>
{
    DenseVector()                                           = delete;
    DenseVector(const DenseVector &rhs)                     = delete;
    DenseVector& operator=(const DenseVector &rhs)          = delete;
    DenseVector& operator=(DenseVector &&rhs)               = delete;

    public:
        using is_DenseVector = std::true_type;

        DenseVector(std::size_t length)
            : Mixin_VectorDim(length, 1),
              Mixin_Conj<false>(false),
              Mixin_Data<T, false>(length)
        {
        }

        DenseVector(DenseVector &&rhs)                          = default;
        ~DenseVector()                                          = default;
};

template <typename T>
class DenseVectorView
    : public Mixin_VectorDim,
      public Mixin_Conj<false>,
      public Mixin_Data<T, true>,
      public Mixin_View<DenseVectorView<T>>,
      public Mixin_VectorViewSelect<DenseVectorView<T>>,
      public Mixin_VectorElementAccess<DenseVectorView<T>>
{
    DenseVectorView()                                       = delete;
    DenseVectorView(const DenseVectorView &rhs)             = delete;
    DenseVectorView& operator=(const DenseVectorView &rhs)  = delete;
    DenseVectorView& operator=(DenseVectorView &&rhs)       = delete;

    public:
        using is_DenseVector = std::true_type;

        DenseVectorView(std::size_t length, bool conj, T *data,
                        std::ptrdiff_t inc)
            : Mixin_VectorDim(length, inc),
              Mixin_Conj<false>(conj),
              Mixin_Data<T, true>(data)
        {
        }

        DenseVectorView(DenseVectorView &&rhs)              = default;
        ~DenseVectorView()                                  = default;
};

template <typename T>
class DenseVectorConstView
    : public Mixin_VectorDim,
      public Mixin_Conj<true>,
      public Mixin_Data<const T, true>,
      public Mixin_View<DenseVectorConstView<T>>,
      public Mixin_VectorViewSelect<DenseVectorConstView<T>>,
      public Mixin_VectorElementAccess<DenseVectorConstView<T>>
{
    DenseVectorConstView()                                           = delete;
    DenseVectorConstView(const DenseVectorConstView &rhs)            = delete;
    DenseVectorConstView& operator=(const DenseVectorConstView &rhs) = delete;
    DenseVectorConstView& operator=(DenseVectorConstView &&rhs)      = delete;

    public:
        using is_DenseVector = std::true_type;

        DenseVectorConstView(std::size_t length, bool conj, const T *data,
                             std::ptrdiff_t inc)
            : Mixin_VectorDim(length, inc),
              Mixin_Conj<true>(conj),
              Mixin_Data<const T, true>(data)
        {
        }

        DenseVectorConstView(DenseVectorConstView &&rhs)    = default;
        ~DenseVectorConstView()                             = default;

};

/// Functions to create a view
/// --------------------------
///
template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
auto
view_create(const VectorX<T> &x)
{
    return DenseVectorConstView<T>(x.length(), x.conj(), x.data(), x.inc());
}

// allowing a chain like x.constView().view()
template <typename T>
auto
view_create(DenseVectorConstView<T> &x)
{
    return DenseVectorConstView<T>(x.length(), x.conj(), x.data(), x.inc());
}

template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
auto
view_create(VectorX<T> &x)
{
    return DenseVectorView<T>(x.length(), x.conj(), x.data(), x.inc());
}

/// Functions to select a sub view
/// ------------------------------
///
/// - Create view with stride 1 for a dense vector.  Starting at index i
template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
auto
view_select(VectorX<T> &&x, std::size_t i)
{
    assert(i<=x.length());

    return VectorX<T>(x.length()-i, x.conj(), x.data()+i*x.inc(), x.inc());
}

/// - Create view with stride 1 for a dense vector
template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
auto
view_select(VectorX<T> &&x, std::size_t i, std::size_t length)
{
    assert(i+length<=x.length());

    return VectorX<T>(length, x.conj(), x.data()+i*x.inc(), x.inc());
}

/// - Create view with arbitrary stride for a dense vector
template <template<typename> class VectorX, typename T,
          Require< Dense<VectorX<T>> > = true>
auto
view_select(VectorX<T> &&x, std::size_t i, std::size_t length,
            std::ptrdiff_t stride)
{
    assert(i+length<=x.length());
    assert(stride!=0);

    auto data = x.data() + x.inc()*(stride>0 ? i : i+length-1);
    length    = (length+std::abs(stride)-1) / std::abs(stride);

    return VectorX<T>(length, x.conj(), data, stride*x.inc());
}


} } // namespace matvec, hpc

#endif // HPC_MATVEC_DENSEVECTOR_HPP
