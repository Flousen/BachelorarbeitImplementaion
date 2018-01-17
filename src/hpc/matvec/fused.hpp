#ifndef HPC_MATVEC_FUSED_HPP
#define HPC_MATVEC_FUSED_HPP

#include <cassert>
#include <cstddef>
#include <type_traits>

#include <hpc/matvec/densevector.hpp>
#include <hpc/matvec/gematrix.hpp>
#include <hpc/matvec/traits.hpp>

namespace hpc { namespace matvec {

template <std::size_t bf>
struct fused
{
    template <typename Alpha,
              typename T, template<typename> class MatrixA,
                          template<typename> class VectorX,
                          typename VectorY,
              Require< Ge<MatrixA<T>>,
                       Dense<VectorX<T>>,
                       Dense<VectorY>
                     > = true>
    static void
    dot(const Alpha &alpha, const MatrixA<T> &A, const VectorX<T> &x,
         VectorY &&y)
    {
        assert(A.numRows()==bf);
        assert(A.numRows()==y.length() && A.numCols()==x.length());

        T tmp[bf];
        for (std::size_t i=0; i<bf; ++i) {
            tmp[i] = 0;
        }
        for (std::size_t j=0; j<A.numCols(); ++j) {
            for (std::size_t i=0; i<bf; ++i) {
                tmp[i] += A(i,j)*x(j);
            }
        }
        for (std::size_t i=0; i<bf; ++i) {
            y(i) += alpha*tmp[i];
        }
    }

    template <typename Alpha,
              typename T, template<typename> class MatrixA,
                          template<typename> class VectorX,
                          typename VectorY,
              Require< Ge<MatrixA<T>>,
                       Dense<VectorX<T>>,
                       Dense<VectorY>
                     > = true>
    static void
    axpy(const Alpha &alpha, const MatrixA<T> &A, const VectorX<T> &x,
         VectorY &&y)
    {
        assert(A.numCols()==bf);
        assert(A.numRows()==y.length() && A.numCols()==x.length());

        for (std::size_t i=0; i<A.numRows(); ++i) {
            T tmp = 0;
            for (std::size_t j=0; j<bf; ++j) {
                tmp += A(i,j)*x(j);
            }
            y(i) += alpha*tmp;
        }
    }
};


} } // namespace matvec, hpc

#endif // HPC_MATVEC_DOT_HPP
