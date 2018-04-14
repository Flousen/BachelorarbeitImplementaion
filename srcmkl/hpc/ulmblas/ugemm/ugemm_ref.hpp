#ifndef HPC_ULMBLAS_UKERNEL_UGEMM_REF_HPP
#define HPC_ULMBLAS_UKERNEL_UGEMM_REF_HPP

namespace hpc { namespace ulmblas {

template <typename T, std::size_t MR, std::size_t NR>
void
ugemm_ref(std::size_t k, T alpha,
          const T *A, const T *B,
          T beta,
          T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
          const T *, const T *)
{
    T AB[MR*NR];

    for (std::size_t i=0; i<MR*NR; ++i) {
        AB[i] = 0;
    }
    for (std::size_t l=0; l<k; ++l) {
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR; ++j) {
                AB[i*NR + j] += A[i]*B[j];
            }
        }
        A += MR;
        B += NR;
    }
    // Yeah, this is unnecessary if (alpha==0). But ok ...
    for (std::size_t i=0; i<MR*NR; ++i) {
        AB[i] *= alpha;
    }
    // This check for beta is really necessary
    if (beta!=T(0)) {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] *= beta;
                C[i*incRowC+j*incColC] += AB[i*NR+j];
            }
        }
    } else {
        for (std::size_t j=0; j<NR; ++j) {
            for (std::size_t i=0; i<MR; ++i) {
                C[i*incRowC+j*incColC] = AB[i*NR+j];
            }
        }
    }
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_UKERNEL_UGEMM_REF_HPP
