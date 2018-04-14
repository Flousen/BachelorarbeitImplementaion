#ifndef HPC_TOOLS_CONJUGATE_HPP
#define HPC_TOOLS_CONJUGATE_HPP

#include <complex>

namespace hpc { namespace tools {

template <typename T>
T
conjugate(T &&x, bool)
{
    return x;
}

template <typename T>
std::complex<T>
conjugate(const std::complex<T> &x, bool conj)
{
    return conj ? std::conj(x) : x;
}

template <bool conj>
struct static_conjugate
{
    template <typename T>
    static T &
    apply(T &x)
    {
        return x;
    }
};

template <>
struct static_conjugate<true>
{
    template <typename T>
    static T &
    apply(T &x)
    {
        return x;
    }

    template <typename T>
    static std::complex<T>
    apply(std::complex<T> &x)
    {
        return std::conj(x);
    }

    template <typename T>
    static std::complex<T>
    apply(const std::complex<T> &x)
    {
        return std::conj(x);
    }
};

} } // namespace tools, hpc

#endif // HPC_TOOLS_CONJUGATE_HPP
