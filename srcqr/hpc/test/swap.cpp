#include <printf.hpp>
#include <hpc/matvec.hpp>

int
main()
{
    using namespace hpc::matvec;

    GeMatrix<double>    A(4,6);

    apply(A, [&A](std::size_t i, std::size_t j)
    {
        A(i,j) = i*A.numCols()+j+1;
    });

    DenseVector<std::size_t> p(A.numRows());
    for (std::size_t i=0; i<p.length(); ++i) {
        p(i) = i%2==0 ? i : p.length()-1;
    }

    fmt::printf("A = \n"); print(A);
    fmt::printf("p = \n"); print(p);

    swap(p, 0, p.length()-1, A);
    fmt::printf("P*A = \n"); print(A);

    swap(p, p.length()-1, 0, A);
    fmt::printf("P^{-1}*A = \n"); print(A);
}
