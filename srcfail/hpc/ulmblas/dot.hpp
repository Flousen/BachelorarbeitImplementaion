#ifndef HPC_ULMBLAS_DOT_HPP
#define HPC_ULMBLAS_DOT_HPP

#include <cstddef>
#include <type_traits>
#include <hpc/tools/conjugate.hpp>

namespace hpc { namespace ulmblas {

template <typename TX, typename TY>
typename std::common_type<TX,TY>::type
dot(std::size_t n,
    bool conjX, const TX *x, std::ptrdiff_t incX,
    bool conjY, const TY *y, std::ptrdiff_t incY)
{
    using T = typename std::common_type<TX,TY>::type;

    T result = T(0);

    for (std::size_t i=0; i<n; ++i) {
        result += T(tools::conjugate(x[i*incX], conjX))
                 *T(tools::conjugate(y[i*incY], conjY));
    }
    return result;
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_DOT_HPP
