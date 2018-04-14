#ifndef HPC_MATVEC_IAMAX_HPP
#define HPC_MATVEC_IAMAX_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <typename T, template<typename> class VectorX,
          Require< Dense<VectorX<T>> > = true>
std::size_t
iamax(const VectorX<T> &x)
{
    std::size_t iAbsMax = 0;
    T absMax = std::abs(x(0));

    for (std::size_t i=0; i<x.length(); ++i) {
        if (std::abs(x(i)) > absMax) {
            iAbsMax = i;
            absMax = std::abs(x(i));
        }
    }
    return iAbsMax;
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_IAMAX_HPP
