#ifndef HPC_ULMBLAS_AXPY_HPP
#define HPC_ULMBLAS_AXPY_HPP

#include <cstddef>
#include <hpc/tools/conjugate.hpp>

namespace hpc { namespace ulmblas {

template <typename Alpha, typename TX, typename TY>
void
axpy(std::size_t n, const Alpha &alpha,
     bool conjX, const TX *x, std::ptrdiff_t incX,
     TY *y, std::ptrdiff_t incY)
{
    if (alpha==Alpha(0)) {
        return;
    }
    for (std::size_t i=0; i<n; ++i) {
        y[i*incY] += TY(alpha)*TY(tools::conjugate(x[i*incX], conjX));
    }
}

template <typename Alpha, typename TA, typename TB>
void
geaxpy(std::size_t m, size_t n, const Alpha &alpha,
       bool conjA, const TA *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
       TB *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB)
{
    if (m==0 || n==0 || alpha==Alpha(0)) {
        return;
    }
    // if B is row major:   B^T <- alpha*A^T + B^T
    if (incRowB>incColB) {
        geaxpy(n, m, alpha, conjA, A, incColA, incRowA, B, incColB, incRowB);
        return;
    }
    // B is col major:
    for (std::size_t j=0; j<n; ++j) {
        for (std::size_t i=0; i<m; ++i) {
            B[i*incRowB+j*incColB]
                += TB(alpha)*TB(tools::conjugate(A[i*incRowA+j*incColA],conjA));
        }
    }
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_AXPY_HPP
