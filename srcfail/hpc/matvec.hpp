#define TULM

#ifndef HPC_MATVEC_HPP
#define HPC_MATVEC_HPP

#include <hpc/matvec/gematrix.hpp>
#include <hpc/matvec/trmatrix.hpp>
#include <hpc/matvec/densevector.hpp>

#include <hpc/matvec/lu.hpp>
#include <hpc/matvec/qr.hpp>

#include <hpc/matvec/apply.hpp>
#include <hpc/matvec/axpy.hpp>
#include <hpc/matvec/copy.hpp>
#include <hpc/matvec/dot.hpp>
#include <hpc/matvec/iamax.hpp>
#include <hpc/matvec/print.hpp>
#include <hpc/matvec/scal.hpp>
#include <hpc/matvec/swap.hpp>

#ifdef ULM 

#include <hpc/matvec/rank1.hpp>
#include <hpc/matvec/mm.hpp>
#include <hpc/matvec/mv.hpp>
#include <hpc/matvec/sv.hpp>

#else

#include <hpc/mklblas/mm.hpp>
#include <hpc/mklblas/mv.hpp>
#include <hpc/mklblas/sm.hpp>
#include <hpc/mklblas/sv.hpp>
#include <hpc/mklblas/trmm.hpp>
#include <hpc/mklblas/trmv.hpp>

#endif


#endif // HPC_MATVEC_HPP
