#include <printf.hpp>
#include <hpc/matvec.hpp>

int
main()
{
    using namespace hpc::matvec;

    {
        GeMatrix<double>    A(4,6);         // col major by default

        auto A_2 = A.dim(A.numRows(),2);

        apply(A_2, [&A_2](std::size_t i, std::size_t j)
        {
            A_2(i,j) = i*A_2.numCols()+j+1;
        });

        fmt::printf("Col major:\n\n");
        fmt::printf("A = \n"); print(A);

        rank1(2, A.col(0,0), A.col(0,1), A.block(0,2));

        fmt::printf("A = \n"); print(A);
    }

    {
        GeMatrix<double>    A(4,6, Order::RowMajor);

        auto A_2 = A.dim(A.numRows(),2);

        apply(A_2, [&A_2](std::size_t i, std::size_t j)
        {
            A_2(i,j) = i*A_2.numCols()+j+1;
        });

        fmt::printf("Row major:\n\n");
        fmt::printf("A = \n"); print(A);

        rank1(2, A.col(0,0), A.col(0,1), A.block(0,2));

        fmt::printf("A = \n"); print(A);
    }
}
