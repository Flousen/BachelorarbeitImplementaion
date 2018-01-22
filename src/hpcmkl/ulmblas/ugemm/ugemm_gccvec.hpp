#ifndef HPC_ULMBLAS_UKERNEL_UGEMM_GCCVEC_HPP
#define HPC_ULMBLAS_UKERNEL_UGEMM_GCCVEC_HPP

namespace hpc { namespace ulmblas {

template <typename T, std::size_t MR, std::size_t NR, std::size_t vec_bits>
void
ugemm_gccvec(std::size_t k, T alpha,
             const T *A, const T *B,
             T beta,
             T *C, std::ptrdiff_t incRowC, std::ptrdiff_t incColC,
             const T *, const T *)
{
    constexpr std::size_t vec_bytes = vec_bits/8;
    constexpr std::size_t vec_dbls  = vec_bytes/sizeof(T);
    constexpr std::size_t NR_       = NR / vec_dbls;

    typedef T vec __attribute__((vector_size (vec_bytes)));

    A = (const T*) __builtin_assume_aligned (A, vec_bytes);
    B = (const T*) __builtin_assume_aligned (B, vec_bytes);

    vec AB[MR*NR_] = {};

    for (std::size_t l=0; l<k; ++l) {
        const vec *b = (const vec *)B;
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR_; ++j) {
                AB[i*NR_ + j] += A[i]*b[j];
            }
        }
        A += MR;
        B += NR;
    }
    for (std::size_t i=0; i<MR; ++i) {
        for (std::size_t j=0; j<NR_; ++j) {
            AB[i*NR_+j] *= alpha;
        }
    }
    if (beta!=T(0)) {
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR_; ++j) {
                const T *p = (const T *) &AB[i*NR_+j];
                for (std::size_t j0=0; j0<vec_dbls; ++j0) {
                    C[i*incRowC+(j*vec_dbls+j0)*incColC] *= beta;
                    C[i*incRowC+(j*vec_dbls+j0)*incColC] += p[j0];
                }
            }
        }
    } else {
        for (std::size_t i=0; i<MR; ++i) {
            for (std::size_t j=0; j<NR_; ++j) {
                const T *p = (const T *) &AB[i*NR_+j];
                for (std::size_t j0=0; j0<vec_dbls; ++j0) {
                    C[i*incRowC+(j*vec_dbls+j0)*incColC] = p[j0];
                }
            }
        }
    }
}

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_UKERNEL_UGEMM_GCCVEC_HPP
