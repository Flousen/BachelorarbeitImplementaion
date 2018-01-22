#ifndef HPC_MATVEC_LU_HPP
#define HPC_MATVEC_LU_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/iamax.hpp>
#include <hpc/matvec/rank1.hpp>
#include <hpc/matvec/scal.hpp>
#include <hpc/matvec/sv.hpp>
#include <hpc/matvec/sm.hpp>
#include <hpc/matvec/swap.hpp>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/traits.hpp>
#include <hpc/ulmblas/axpy.hpp>

namespace hpc { namespace matvec {

template <typename MatrixA, typename VectorPiv,
          Require< Ge<MatrixA>,
                   Dense<VectorPiv>
                 > = true>
std::ptrdiff_t
lu_unblk_var1(MatrixA &&A, VectorPiv &&piv)
{
    using T = ElementType<MatrixA>;

    std::size_t m  = A.numRows();
    std::size_t n  = A.numCols();
    std::size_t mn = std::min(m,n);

    assert(piv.length()>=mn);

    for (std::size_t j=0; j<mn; ++j) {
        // pivoting
        auto jp = j + iamax(A.col(j,j));
        if (jp!=j) {
            swap(A.row(j,0), A.row(jp,0));
        }
        piv(j) = jp;
        if (A(j,j)==T(0)) {
            return j;
        }

        // apply gauss
        scal(1/A(j,j), A.col(j+1,j));
        rank1(T(-1), A.col(j+1,j), A.row(j,j+1),
              A.block(j+1,j+1));
    }
    return -1;
}

template <typename MatrixA, typename VectorPiv,
          Require< Ge<MatrixA>,
                   Dense<VectorPiv>
                 > = true>
std::ptrdiff_t
lu_unblk_var2(MatrixA &&A, VectorPiv &&piv)
{
    using T = ElementType<MatrixA>;

    std::size_t m  = A.numRows();
    std::size_t n  = A.numCols();
    std::size_t mn = std::min(m,n);

    assert(piv.length()>=mn);

    for (std::size_t j=0; j<mn; ++j) {
        sv(A.dim(j,j).view(UpLo::LowerUnit), A.col(0,j).dim(j));
        mv(T(-1),
           A.block(j,0).dim(m-j,j), A.col(0,j).dim(j),
           T(1),
           A.col(j,j));

        // pivoting
        auto jp = j + iamax(A.col(j,j));
        if (jp!=j) {
            swap(A.row(j,0), A.row(jp,0));
        }
        piv(j) = jp;
        if (A(j,j)==T(0)) {
            return j;
        }

        if (j+1<m) {
            scal(1/A(j,j), A.col(j+1,j));
        }
    }
    for (std::size_t j=mn; j<n; ++j) {
        sv(A.dim(m,m).view(UpLo::LowerUnit), A.col(0,j).dim(m));
    }
    return -1;
}

template <typename MatrixA, typename VectorPiv,
          Require< Ge<MatrixA>,
                   Dense<VectorPiv>
          > = true>
std::ptrdiff_t
lu_blk_var1(MatrixA &&A, VectorPiv &&piv)
{
    using T = ElementType<MatrixA>;

    std::size_t m = A.numRows();
    std::size_t n = A.numCols();
    std::size_t mn = std::min(m,n);

    assert(piv.length()>=mn);

    constexpr std::size_t bs = 64;

    if (bs<=1 || bs>mn) {
        return lu_unblk_var2(A, piv);
    } else {
        std::ptrdiff_t info = -1;

        for (std::size_t j=0; j<mn; j+=bs) {
            std::size_t jb = std::min(mn-j, bs);

            auto info_ = lu_unblk_var2(A.block(j,j).dim(m-j,jb), piv.block(j));
            if (info==-1 && info_>-1) {
                info = j + info_;
            }
            for (std::size_t k=j; k<j+jb; ++k) {
                piv(k) += j;
            }
            swap(piv, j, j+jb-1, A.dim(m,j));
            if (j+jb<n) {
                swap(piv, j, j+jb-1, A.block(0,j+jb).dim(m,n-(j+jb)));
                sm(T(1),
                   A.block(j,j).dim(jb,jb).view(matvec::UpLo::LowerUnit),
                   A.block(j,j+jb).dim(jb, n-(j+jb)));
            }
            if (j+jb<n && j+jb<m) {
                mm(T(-1),
                   A.block(j+jb,j).dim(m-(j+jb),jb),
                   A.block(j,j+jb).dim(jb,n-(j+jb)),
                   T(1),
                   A.block(j+jb,j+jb));
            }
        }
        return info;
    }
}

template <typename MatrixA, typename VectorPiv,
          Require< Ge<MatrixA>,
                   Dense<VectorPiv>
          > = true>
std::ptrdiff_t
lu_blk_var2(MatrixA &&A, VectorPiv &&piv)
{
    //using T = ElementType<MatrixA>;

    std::size_t m = A.numRows();
    std::size_t n = A.numCols();
    std::size_t mn = std::min(m,n);

    assert(piv.length()>=mn);

    // TODO: your code for quiz06
    return -1;
}


} } // namespace matvec, hpc

#endif // HPC_MATVEC_LU_HPP
