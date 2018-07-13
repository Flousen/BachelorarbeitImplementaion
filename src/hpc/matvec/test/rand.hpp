#ifndef HPC_MATVEC_TEST_RAND_HPP
#define HPC_MATVEC_TEST_RAND_HPP

#include <complex>
#include <random>

namespace hpc { namespace matvec { namespace test {

template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<T>> > = true>
void
rand(MatrixA<T> &A)
{
    std::random_device                  random;
    std::mt19937                        mt(random());
    std::uniform_real_distribution<T>   uniform(-1,1);

    apply(A, [&](std::size_t i, std::size_t j)
    {
        A(i,j) = uniform(mt);
    });
}

//template <template<typename> class MatrixA, typename T,
//          Require< Ge<MatrixA<T>> > = true>
//void
//init(MatrixA<T> &A)
//{
//  for( T p = 0 , size_t i=0; i < A.numRows; ++i, ++p){
//    for( size_t j=0; j < A.numCols; ++j){
//      A(i,j) = p;
//    }
//  }
//}


template <template<typename> class MatrixA, typename T,
          Require< Ge<MatrixA<std::complex<T>>> > = true>
void
rand(MatrixA<std::complex<T>> &A)
{
    std::random_device                  random;
    std::mt19937                        mt(random());
    std::uniform_real_distribution<T>   uniform(-1,1);

    apply(A, [&](std::size_t i, std::size_t j)
    {
        A(i,j) = std::complex<T>(uniform(mt), uniform(mt));
    });
}

} } } // namespace test, matvec, hpc

#endif // HPC_MATVEC_TEST_RAND_HPP
