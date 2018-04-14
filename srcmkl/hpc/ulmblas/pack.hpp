#ifndef HPC_ULMBLAS_PACK_HPP
#define HPC_ULMBLAS_PACK_HPP

#include <cstddef>
#include <hpc/tools/conjugate.hpp>

namespace hpc { namespace ulmblas {

template <typename T, std::size_t MR>
void
pack_A(std::size_t M, std::size_t K, bool conjA,
       const T *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
       T *p)
{
    std::size_t mp = (M + MR - 1) / MR;

    if (incRowA<incColA) {
        for (std::size_t J=0; J<K; ++J) {
            for (std::size_t I=0; I<MR*mp; ++I) {
                std::size_t mu = MR*K*(I/MR) + J*MR + (I % MR);

                p[mu] = (I<M) ? tools::conjugate(A[I*incRowA+J*incColA], conjA)
                              : T(0);
            }
        }
    } else {
        for (std::size_t I=0; I<MR*mp; ++I) {
            for (std::size_t J=0; J<K; ++J) {
                std::size_t mu = MR*K*(I/MR) + J*MR + (I % MR);

                p[mu] = (I<M) ? tools::conjugate(A[I*incRowA+J*incColA], conjA)
                              : T(0);
            }
        }
    }
}

template <typename T, std::size_t NR>
void
pack_B(std::size_t K, std::size_t N, bool conjB,
       const T *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
       T *p)
{
    pack_A<T, NR>(N, K, conjB, B, incColB, incRowB, p);
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_PACK_HPP
