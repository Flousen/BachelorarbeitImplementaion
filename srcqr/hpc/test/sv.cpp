#include <printf.hpp>
#include <hpc/matvec.hpp>

int
main()
{
    using namespace hpc::matvec;

    {
        std::size_t m = 4;

        GeMatrix<double>    A(m, m+2);         // col major by default

        apply(A, [=,&A](std::size_t i, std::size_t j)
        {
            // initialize only lower triangular part and
            // column with index j = A.numRows()
            A(i,j) = (i==j) ? 1 :
                     (i>j)  ? i*m+j :
                     (j==m) ? i+1 :
                     0;
        });

        fmt::printf("Col major:\n\n");
        mv(1, A.dim(m,m), A.col(0,m), 0, A.col(0,m+1));

        // modify diagonal
        for (std::size_t i=0; i<m; ++i) {
            A(i,i) = 42;
        }
        fmt::printf("A = \n"); print(A);

        // triangular solver should ignore the entries on the diagonal
        sv(A.dim(m,m).view(UpLo::LowerUnit), A.col(0,m+1));
        // A.col(0,m+1) and A.col(0,m) should now be equal
        fmt::printf("A = \n"); print(A);
    }

    {
        std::size_t m = 4;

        GeMatrix<double>    A(m, m+2, Order::RowMajor);

        apply(A, [=,&A](std::size_t i, std::size_t j)
        {
            A(i,j) = (i==j) ? 1 :
                     (i>j)  ? i*m+j :
                     (j==m) ? i+1 :
                     0;
        });

        fmt::printf("Row major:\n\n");
        mv(1, A.dim(m,m), A.col(0,m), 0, A.col(0,m+1));

        for (std::size_t i=0; i<m; ++i) {
            A(i,i) = 42;
        }
        fmt::printf("A = \n"); print(A);

        sv(A.dim(m,m).view(UpLo::LowerUnit), A.col(0,m+1));
        fmt::printf("A = \n"); print(A);
    }
}
