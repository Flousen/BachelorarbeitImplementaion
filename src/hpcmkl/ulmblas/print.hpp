#ifndef HPC_ULMBLAS_PRINT_HPP
#define HPC_ULMBLAS_PRINT_HPP

#include <complex>
#include <printf.hpp>
#include <hpc/tools/conjugate.hpp>

namespace hpc { namespace ulmblas {

/// - Print function for vectors
template <typename TX>
void
print(std::size_t n, bool, const TX *x, std::ptrdiff_t incX,
      const char *format = " %6.2lf")
{
    for (std::size_t i=0; i<n; ++i) {
        fmt::printf(format, x[i*incX]);
    }
    fmt::printf("\n");
}
///-

template <typename TX>
void
print(std::size_t n, bool conjX, const std::complex<TX> *x, std::ptrdiff_t incX,
      const char *format = " (%6.2f%+6.2fi)")
{
    for (std::size_t i=0; i<n; ++i) {
        fmt::printf(format, x[i*incX].real(),
                    tools::conjugate(x[i*incX], conjX).imag());
    }
    fmt::printf("\n");
}

/// - Print function for matrices
template <typename TX>
void
geprint(std::size_t m, std::size_t n,
        bool conjX, const TX *X, std::ptrdiff_t incRowX, std::ptrdiff_t incColX,
        const char *format = " %8.2lf")
{
    for (std::size_t i=0; i<m; ++i) {
        print(n, conjX, &X[i*incRowX], incColX, format);
    }
    fmt::printf("\n");
}
///-

template <typename TX>
void
geprint(std::size_t m, std::size_t n, bool conjX, const std::complex<TX> *X,
        std::ptrdiff_t incRowX, std::ptrdiff_t incColX,
        const char *format = " (%6.2f%+6.2fi)")
{
    for (std::size_t i=0; i<m; ++i) {
        print(n, conjX, &X[i*incRowX], incColX, format);
    }
    fmt::printf("\n");
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_PRINT_HPP
