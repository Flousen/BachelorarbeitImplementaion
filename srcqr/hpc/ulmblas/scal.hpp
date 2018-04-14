#ifndef HPC_ULMBLAS_SCAL_HPP
#define HPC_ULMBLAS_SCAL_HPP

#include <cstddef>

namespace hpc { namespace ulmblas {

template <typename Alpha, typename TX>
void
scal(std::size_t n, const Alpha & alpha,
     TX *x, std::ptrdiff_t incX)
{
    if (alpha!=Alpha(0)) {
        for (std::size_t i=0; i<n; ++i) {
            x[i*incX] *= TX(alpha);
        }
    } else {
        for (std::size_t i=0; i<n; ++i) {
            x[i*incX] = Alpha(0);
        }
    }
}

template <typename Alpha, typename TA>
void
gescal(std::size_t m, size_t n,
       const Alpha & alpha,
       TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA)
{
    if (m==0 || n==0 || alpha==Alpha(1)) {
        return;
    }
    // A is row major: scale A^T
    if (incRowA>incColA) {
        gescal(n, m, alpha, A, incColA, incRowA);
        return;
    }
    // A is col major:
    if (alpha!=Alpha(0)) {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] *= TA(alpha);
            }
        }
    } else {
        for (std::size_t j=0; j<n; ++j) {
            for (std::size_t i=0; i<m; ++i) {
                A[i*incRowA+j*incColA] = TA(0);
            }
        }
    }
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_COPY_HPP
