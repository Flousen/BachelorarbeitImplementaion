#ifndef HPC_CONFIG_HPP
#define HPC_CONFIG_HPP

#include <cassert>
#include <cstdlib>

namespace hpc {

template <typename T>
T
assertEqual_(const char *, int, T &&x)
{
    return x;
}

template <typename T, typename... Args>
T
assertEqual_(const char *file, int line, T &&x, Args... args)
{
    if (x!=assertEqual_(file, line, args...)) {
        fmt::printf("assertEqual failed in file %s line %d\n", file, line);
        abort();
    }
    return x;
}

#define assertEqual(...)    assertEqual_(__FILE__, __LINE__, __VA_ARGS__)

} // namespace hpc

#endif // HPC_CONFIG_HPP
