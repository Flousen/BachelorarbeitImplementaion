#ifndef HPC_ULMBLAS_CONFIG_HPP
#define HPC_ULMBLAS_CONFIG_HPP

#include <cstdlib>
#include <complex>
#include <map>
#include <string>

// Include mgemm, pack_A and pack_B
#include <hpc/ulmblas/mgemm.hpp>
#include <hpc/ulmblas/pack.hpp>

// Include potential micro kernels
#include <hpc/ulmblas/ugemm/avx_cugemm_8x4.hpp>
#include <hpc/ulmblas/ugemm/avx_dugemm_8x4.hpp>
#include <hpc/ulmblas/ugemm/avx_sugemm_8x8.hpp>
#include <hpc/ulmblas/ugemm/avx_zugemm_4x4.hpp>
#include <hpc/ulmblas/ugemm/ugemm_gccvec.hpp>
#include <hpc/ulmblas/ugemm/ugemm_ref.hpp>

namespace hpc { namespace ulmblas {

// Stuff that actually interacts with the OS
namespace system {

// Possible configurations
enum Config {
    Undefined   = 0,
    Generic     = 1,
    SSE         = 2,
    AVX         = 3
};

Config
getConfig()
{
    static Config config = Config::Undefined;
    static std::map<std::string, Config> configMap = {
            {"GENERIC",  Generic},
            {"SSE",      SSE},
            {"AVX",      AVX}
        };

    if (config==Config::Undefined) {
        const char *env = std::getenv("ULMBLAS");
        std::string sel = (env) ? std::string(env) : std::string();
        if (configMap.count(sel)>0) {
            config = configMap[sel];
        } else {
            config = Config::Generic;
        }
    }
    return config;
}

} // namespace system

// Function type for a GEMM macro kernel
template <typename T>
using MGemm = void (*)(std::size_t, std::size_t, std::size_t,
                       T, const T *, const T *, T,
                       T *, std::ptrdiff_t, std::ptrdiff_t);
// Function type for a packing
template <typename T>
using Pack  = void (*)(std::size_t, std::size_t, bool,
                       const T *, std::ptrdiff_t, std::ptrdiff_t,
                       T *);

//
// Select reference implementation in the general case
//
template <typename T>
struct GemmParameter
{
    // Parameters for gemm
    std::size_t MC, NC, KC;
    std::size_t alignment;
    std::size_t extra_A, extra_B;
    MGemm<T> mgemm;
    Pack<T>  pack_A, pack_B;

    GemmParameter(std::size_t m, std::size_t n, std::size_t k)
    {
        MC = 256;
        NC = 2048;
        KC = 256;
        alignment = 0;
        extra_A   = 0;
        extra_B   = 0;

        mgemm  = ulmblas::mgemm<T,4,64,ugemm_ref<T,4,64> >;
        pack_A = ulmblas::pack_A<T,4>;
        pack_B = ulmblas::pack_B<T,64>;
    }
};

template <>
struct GemmParameter<float>
{
    // Parameters for gemm
    std::size_t MC, NC, KC;
    std::size_t alignment;
    std::size_t extra_A, extra_B;

    MGemm<float> mgemm;
    Pack<float>  pack_A, pack_B;

    GemmParameter(std::size_t /*m*/, std::size_t /*n*/, std::size_t /*k*/)
    {
        extra_A   = 0;
        extra_B   = 0;
        switch (system::getConfig()) {
            case system::AVX:
                MC = 128;
                NC = 2048;
                KC = 384;
                alignment = 32;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 8;

                mgemm  = ulmblas::mgemm<float, 8,8, sugemm_asm_8x8>;
                pack_A = ulmblas::pack_A<float, 8>;
                pack_B = ulmblas::pack_B<float, 8>;
                break;

            case system::SSE:
                MC = 256;
                NC = 2048;
                KC = 256;
                alignment = 16;

                mgemm  = ulmblas::mgemm<float, 4,4,
                                        ugemm_gccvec<float, 4,4,128>>;
                pack_A = ulmblas::pack_A<float, 4>;
                pack_B = ulmblas::pack_B<float, 4>;
                break;

            default:
                MC = 256;
                NC = 2048;
                KC = 256;
                alignment = 0;

                mgemm  = ulmblas::mgemm<float, 4,64,
                                        ugemm_ref<float, 4,64> >;
                pack_A = ulmblas::pack_A<float, 4>;
                pack_B = ulmblas::pack_B<float, 64>;
        }
        // Fine tuning MC, NC, KC (depending on m, n, k) could be done here ...

    }
};

template <>
struct GemmParameter<double>
{
    // Parameters for gemm
    std::size_t MC, NC, KC;
    std::size_t alignment;
    std::size_t extra_A, extra_B;

    MGemm<double> mgemm;
    Pack<double>  pack_A, pack_B;

    GemmParameter(std::size_t /*m*/, std::size_t /*n*/, std::size_t /*k*/)
    {
        extra_A   = 0;
        extra_B   = 0;
        switch (system::getConfig()) {
            case system::AVX:
                MC = 256;
                NC = 2048;
                KC = 256;
                alignment = 32;
                //extra_A   = 8;
                //extra_B   = 4;

                //mgemm  = ulmblas::mgemm<double, 8,4, dugemm_asm_8x4>;
                //pack_A = ulmblas::pack_A<double, 8>;
                //pack_B = ulmblas::pack_B<double, 4>;
                mgemm  = ulmblas::mgemm<double, 4,8,
                                        ugemm_gccvec<double, 4,8,256>>;
                pack_A = ulmblas::pack_A<double, 4>;
                pack_B = ulmblas::pack_B<double, 8>;
                break;

            case system::SSE:
                MC = 256;
                NC = 2048;
                KC = 256;
                alignment = 16;

                mgemm  = ulmblas::mgemm<double, 4,4,
                                        ugemm_gccvec<double, 4,4,128>>;
                pack_A = ulmblas::pack_A<double, 4>;
                pack_B = ulmblas::pack_B<double, 4>;
                break;

            default:
                MC = 256;
                NC = 2048;
                KC = 256;
                alignment = 0;

                mgemm  = ulmblas::mgemm<double, 4,64,
                                        ugemm_ref<double, 4,64> >;
                pack_A = ulmblas::pack_A<double, 4>;
                pack_B = ulmblas::pack_B<double, 64>;
        }
        // Fine tuning MC, NC, KC could be done here ...

    }
};

template <>
struct GemmParameter<std::complex<float>>
{
    // Parameters for gemm
    std::size_t MC, NC, KC;
    std::size_t alignment;
    std::size_t extra_A, extra_B;

    MGemm<std::complex<float>> mgemm;
    Pack<std::complex<float>>  pack_A, pack_B;

    GemmParameter(std::size_t /*m*/, std::size_t /*n*/, std::size_t /*k*/)
    {
        extra_A   = 0;
        extra_B   = 0;
        switch (system::getConfig()) {
            case system::AVX:
                MC = 96;
                NC = 4096;
                KC = 256;
                alignment = 32;
                alignment = 32;
                extra_A   = 8;
                extra_B   = 4;

                mgemm  = ulmblas::mgemm<std::complex<float>, 8,4, cugemm_asm_8x4>;
                pack_A = ulmblas::pack_A<std::complex<float>, 8>;
                pack_B = ulmblas::pack_B<std::complex<float>, 4>;
                break;


            default:
                MC = 384;
                NC = 4096;
                KC = 384;
                alignment = 0;

                mgemm  = ulmblas::mgemm<std::complex<float>, 4,2,
                                        ugemm_ref<std::complex<float>, 4,2> >;
                pack_A = ulmblas::pack_A<std::complex<float>, 4>;
                pack_B = ulmblas::pack_B<std::complex<float>, 2>;
        }
        // Fine tuning MC, NC, KC could be done here ...

    }
};

template <>
struct GemmParameter<std::complex<double>>
{
    // Parameters for gemm
    std::size_t MC, NC, KC;
    std::size_t alignment;
    std::size_t extra_A, extra_B;

    MGemm<std::complex<double>> mgemm;
    Pack<std::complex<double>>  pack_A, pack_B;

    GemmParameter(std::size_t /*m*/, std::size_t /*n*/, std::size_t /*k*/)
    {
        extra_A   = 0;
        extra_B   = 0;
        switch (system::getConfig()) {
            case system::AVX:
                MC = 64;
                NC = 4096;
                KC = 192;
                alignment = 32;
                alignment = 32;
                extra_A   = 4;
                extra_B   = 4;

                mgemm  = ulmblas::mgemm<std::complex<double>, 4,4, zugemm_asm_4x4>;
                pack_A = ulmblas::pack_A<std::complex<double>, 4>;
                pack_B = ulmblas::pack_B<std::complex<double>, 4>;
                break;


            default:
                MC = 384;
                NC = 4096;
                KC = 384;
                alignment = 0;

                mgemm  = ulmblas::mgemm<std::complex<double>, 4,2,
                                        ugemm_ref<std::complex<double>, 4,2> >;
                pack_A = ulmblas::pack_A<std::complex<double>, 4>;
                pack_B = ulmblas::pack_B<std::complex<double>, 2>;
        }
        // Fine tuning MC, NC, KC could be done here ...

    }
};

} } // namespace ulmblas, hpc

#endif // HPC_ULMBLAS_CONFIG_HPP
