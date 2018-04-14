#ifndef HPC_ULMBLAS_MGEMM_HPP
#define HPC_ULMBLAS_MGEMM_HPP

#include <cstddef>
#include <hpc/ulmblas/axpy.hpp>
#include <hpc/ulmblas/scal.hpp>

namespace hpc { namespace ulmblas {

template <typename T>
using UGemm = void (*)(std::size_t, T,
                       const T *, const T *,
                       T,
                       T *, std::ptrdiff_t, std::ptrdiff_t,
                       const T *, const T *);

template <typename T, std::size_t MR, std::size_t NR, UGemm<T> ugemm>
void
mgemm(std::size_t M, std::size_t N, std::size_t K,
      T alpha,
      const T *A, const T *B,
      T beta,
      T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
{
    std::size_t mp = (M+MR-1) / MR;
    std::size_t np = (N+NR-1) / NR;

    std::size_t mr_ = M % MR;
    std::size_t nr_ = N % NR;

    T C_[MR*NR];

    const T *a_next = A;
    const T *b_next = nullptr;

    for (std::size_t j=0; j<np; ++j) {
        std::size_t nr = (j<np-1 || nr_==0) ? NR
                                            : nr_;
        b_next = &B[j*K*NR];
        for (std::size_t i=0; i<mp; ++i) {
            std::size_t mr = (i<mp-1 || mr_==0) ? MR
                                                : mr_;

            a_next = &A[(i+1)*MR*K];
            if (i==mp-1) {
                a_next = A;
                b_next = &B[(j+1)*K*NR];
                if (j==np-1) {
                    b_next = B;
                }
            }

            if (mr==MR && nr==NR) {
                ugemm(K, alpha,
                      &A[i*MR*K], &B[j*K*NR],
                      beta,
                      &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC,
                      a_next, b_next);
            } else {
                ugemm(K, alpha,
                      &A[i*MR*K], &B[j*K*NR],
                      0,
                      C_, 1, MR,
                      a_next, b_next);
                gescal(mr, nr, beta,
                       &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
                geaxpy(mr, nr, T(1),
                       false, C_, 1, MR,
                       &C[i*MR*incRowC+j*NR*incColC], incRowC, incColC);
            }
        }
    }
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_MGEMM_HPP
