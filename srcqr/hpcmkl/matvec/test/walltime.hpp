#ifndef HPC_MATVEC_TEST_WALLTIME_HPP
#define HPC_MATVEC_TEST_WALLTIME_HPP

#include <chrono>

namespace hpc { namespace matvec { namespace test {

template <typename T>
struct WallTime
{
    void
    tic()
    {
        t0 = std::chrono::high_resolution_clock::now();
    }

    T
    toc()
    {
        using namespace std::chrono;

        elapsed = high_resolution_clock::now() - t0;
        return duration<T,seconds::period>(elapsed).count();
    }

    std::chrono::high_resolution_clock::time_point t0;
    std::chrono::high_resolution_clock::duration   elapsed;
};

} } } // namespace test, matvec, hpc

#endif // HPC_MATVEC_TEST_WALLTIME_HPP
