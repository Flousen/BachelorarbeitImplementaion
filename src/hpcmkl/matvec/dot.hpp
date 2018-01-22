#ifndef HPC_MATVEC_DOT_HPP
#define HPC_MATVEC_DOT_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/dot.hpp>

namespace hpc { namespace matvec {

template <typename T, template<typename> class VectorX,
                      template<typename> class VectorY,
          Require< Dense<VectorX<T>>,
                   Dense<VectorY<T>> > = true>
T
dot(const VectorX<T> &x, const VectorY<T> &y)
{
    assert(x.length()==y.length());
    return ulmblas::dot(x.length(), x.conj(), x.data(), x.inc(),
                        y.conj(), y.data(), y.inc());
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_DOT_HPP
