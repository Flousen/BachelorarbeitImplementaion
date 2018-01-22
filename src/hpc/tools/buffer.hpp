#ifndef HPC_TOOLS_ARRAY_HPP
#define HPC_TOOLS_ARRAY_HPP

#include <cassert>
#include <cstddef>
#include <cstdlib>

namespace hpc { namespace tools {

template <typename T>
T *
malloc_aligned(std::size_t size, std::size_t alignment_)
{
    size_t alignment = std::max(alignment_, alignof(void *));

    size     *= sizeof(T);
    size     += alignment;

    void *ptr  = std::malloc(size);
    void *ptr2 = (void *)(((std::uintptr_t)ptr + alignment) & ~(alignment-1));
    void **vp  = (void**) ptr2 - 1;
    *vp        = ptr;
    return static_cast<T *>(ptr2);
}

template <typename T>
void
free_aligned(T *ptr)
{
    std::free(*((void**)ptr-1));
}

template <typename T>
struct Buffer
{
    Buffer(std::size_t n, std::size_t alignment = alignof(void *))
        : ptr(malloc_aligned<T>(n, alignment))
    {
        if (!ptr) {
            std::abort();
        }
        //fmt::printf("allocated %g\n", ptr);
    }

    Buffer()
        : ptr(nullptr)
    {
    }

    ~Buffer()
    {
        free_aligned(ptr);
        //fmt::printf("released %g\n", ptr);
        //*ptr = T(42);
    }

    Buffer(Buffer &&)           = default;

    Buffer(const Buffer &)      = delete;

    Buffer &
    operator=(const Buffer &)   = delete;

    Buffer &
    operator=(Buffer &&)        = delete;

    T &
    operator[](std::size_t i) const
    {
        return ptr[i];
    }

    T * const ptr;
};

} } // namespace tools, hpc

#endif // HPC_TOOLS_ARRAY_HPP
