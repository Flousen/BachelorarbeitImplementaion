#ifndef HPC_MATVEC_VIEWS_HPP
#define HPC_MATVEC_VIEWS_HPP

#include <cmath>
#include  <utility>

namespace hpc { namespace matvec {

// View type selectors:
enum class Conj         { view };
enum class Trans        { view };
enum class ConjTrans    { view };
enum class RowRange     { view };
enum class ColRange     { view };
enum class Row          { view };
enum class Col          { view };

template <typename T>
auto
constView(const T &x)
{
    return view_create(x);
}

template <typename T>
auto
view(T &&x)
{
    return view_create(x);
}

template <typename T, typename... Args>
auto
constView(const T &x, Args... args)
{
    return view_select(view_create(x), args...);
}

template <typename T, typename... Args>
auto
view(T &&x, Args... args)
{
    return view_select(view_create(x), args...);
}

} } // namespace matvec, hpc

#endif // HPC_MATVEC_VIEWS_HPP
