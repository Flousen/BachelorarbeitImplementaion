#ifndef HPC_ULMBLAS_GEMM_HPP
#define HPC_ULMBLAS_GEMM_HPP

#include <cstddef>
#include <hpc/tools/buffer.hpp>
#include <hpc/ulmblas/ugemm/config.hpp>

namespace hpc { namespace ulmblas {

template <typename T>
void
gemm(std::size_t m, std::size_t n, std::size_t k,
      T alpha,
      bool conjA, const T *A, std::ptrdiff_t incRowA, std::ptrdiff_t incColA,
      bool conjB, const T *B, std::ptrdiff_t incRowB, std::ptrdiff_t incColB,
      T beta,
      T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC)
{
    if (alpha==T(0) || k==0) {
        gescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    GemmParameter<T>   p(m, n, k);
    std::size_t MC = p.MC, NC = p.NC, KC = p.KC;

    std::size_t mb = (m + MC-1)/MC;
    std::size_t nb = (n + NC-1)/NC;
    std::size_t kb = (k + KC-1)/KC;

    std::size_t mc_ = m % MC;
    std::size_t nc_ = n % NC;
    std::size_t kc_ = k % KC;

    tools::Buffer<T> A_(MC*KC+p.extra_A, p.alignment);
    tools::Buffer<T> B_(KC*NC+p.extra_B, p.alignment);

    for (std::size_t j=0; j<nb; ++j) {
        std::size_t N = (j<nb-1 || nc_==0) ? NC
                                           : nc_;
        for (std::size_t l=0; l<kb; ++l) {
            std::size_t K = (l<kb-1 || kc_==0) ? KC
                                               : kc_;
            T beta_ = (l==0) ? beta
                             : 1;
            p.pack_B(K, N, conjB,
                     &B[l*KC*incRowB+j*NC*incColB], incRowB, incColB,
                     B_.ptr);

            for (std::size_t i=0; i<mb; ++i) {
                std::size_t M = (i<mb-1 || mc_==0) ? MC
                                                   : mc_;

                p.pack_A(M, K, conjA,
                         &A[i*MC*incRowA+l*KC*incColA], incRowA, incColA,
                         A_.ptr);

                p.mgemm(M, N, K, alpha, A_.ptr, B_.ptr, beta_,
                        &C[i*MC*incRowC+j*NC*incColC], incRowC, incColC);
            }
        }
    }
}


} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_GEMM_HPP
