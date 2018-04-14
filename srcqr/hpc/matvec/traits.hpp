#ifndef HPC_MATVEC_TRAITS_HPP
#define HPC_MATVEC_TRAITS_HPP

#include <complex>

namespace hpc { namespace matvec {

/// Real

template <typename T>
struct make_real
{
    using type = T;
};

template <typename T>
struct make_real<std::complex<T>>
{
    using type = T;
};

template <typename T>
struct make_real<const std::complex<T>>
{
    using type = const T;
};

template <typename T>
using Real = typename make_real<typename std::remove_reference<T>::type>::type;

template <typename T>
struct make_ElementType
{
};

template <template<typename> class X, typename T>
struct make_ElementType<X<T>>
{
    using type = T;
};

template <typename T>
using ElementType
    = typename make_ElementType<typename std::remove_reference<T>::type>::type;

template <typename TX, typename TY>
using SameElementType
    =  typename std::conditional<
                std::is_same<ElementType<TX>, ElementType<TY>>::value,
                bool, void>::type;


/// Views
template <typename T>
using SubVector = typename T::VectorView;

/// Require
template <typename T>
using Dense
    = std::enable_if_t<std::remove_reference<T>::type::is_DenseVector::value,
                       bool>;

template <typename T>
using Ge
    = std::enable_if_t<std::remove_reference<T>::type::is_GeMatrix::value,
                       bool>;

template <typename T>
using Tr
    = std::enable_if_t<std::remove_reference<T>::type::is_TrMatrix::value,
                       bool>;

template <typename T, typename... Args>
using Require = typename std::common_type<T, Args...>::type;

} } // namespace matvec, hpc

#endif // HPC_MATVEC_TRAITS_HPP
