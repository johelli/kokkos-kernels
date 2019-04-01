/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2019 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact J. Austin Ellis (johelli@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#ifndef KOKKOSDNN_CONV2D_SPEC_HPP_
#define KOKKOSDNN_CONV2D_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include<KokkosDNN_conv2d_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template<class AVT, class FVT, class MVT>
struct conv2d_eti_spec_avail {
  enum : bool { value = false };
};
}
}


//
// Macro for declaration of full specialization availability
// KokkosDNN::Impl::CONV2D.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSDNN_CONV2D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, LAYOUTA, LAYOUTF, LAYOUTM, EXEC_SPACE, MEM_SPACE ) \
    template<> \
    struct gemm_eti_spec_avail< \
         Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR**, LAYOUTM, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
         > { enum : bool { value = true }; };

#define KOKKOSDNN_CONV2D_ETI_SPEC_AVAIL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSDNN_CONV2D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE)

// Include the actual specialization declarations
#include<KokkosDNN_conv2d_tpl_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosDNN_conv2d_eti_spec_avail.hpp>

namespace KokkosDNN {
namespace Impl {

//
// conv2d
//

// Implementation of KokkosDNN::conv2d.
template<class AViewType,
         class FViewType,
         class MViewType,
         bool tpl_spec_avail = gemm_tpl_spec_avail<AViewType, FViewType, MViewType>::value,
         bool eti_spec_avail = gemm_eti_spec_avail<AViewType, FViewType, MViewType>::value
         >
struct CONV2D {
  static void
  conv2d (const AViewType& A,
          const FViewType& F,
          const int stride, 
          const MViewType& M)
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
{
  static_assert (Kokkos::Impl::is_view<AViewType>::value,
                 "AViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<FViewType>::value,
                 "FViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<MViewType>::value,
                 "MViewType must be a Kokkos::View.");
  static_assert (static_cast<int> (AViewType::rank) == 2,
                 "AViewType must have rank 2.");
  static_assert (static_cast<int> (FViewType::rank) == 2,
                 "FViewType must have rank 2.");
  static_assert (static_cast<int> (MViewType::rank) == 2,
                 "MViewType must have rank 2.");

  Kokkos::Profiling::pushRegion(eti_spec_avail?"KokkosDNN::conv2d[ETI]":"KokkosDNN::conv2d[noETI]");
  // Figure out Scalar Types
  typedef typename AViewType::non_const_value_type ScalarA;
  typedef typename FViewType::non_const_value_type ScalarF;
  typedef typename MViewType::non_const_value_type ScalarM;

  // Define Blocking sizes (this will be used for scratch spaces)
  static constexpr int blockA0 = 24;
  static constexpr int blockB1 = 64;
  static constexpr int blockA1 = 
    (sizeof(ScalarA) * blockA0 * 16 + 
     sizeof(ScalarB) * 16 * blockB1 + 
     sizeof(ScalarC) * blockA0 * blockB1 < 24000) ? 
        16 : (sizeof(ScalarA) * blockA0 * 8 + 
              sizeof(ScalarB) * 8 * blockB1 + 
              sizeof(ScalarC) * blockA0 * blockB1 < 24000) ? 
          8 :  (sizeof(ScalarA) * blockA0 * 4 + 
                sizeof(ScalarB) * 4 * blockB1 + 
                sizeof(ScalarC) * blockA0 * blockB1 < 24000) ? 
            4 : 16;

  static constexpr int vector_length = blockB1 / 4;

  // Compute scratch space size
  typedef KokkosBlas::Impl::CONV2DImpl<typename MViewType::execution_space, 
                                       AViewType, FViewType, MViewType, 
                                       blockA0, blockA1, blockB1, 0, 0> gemm_dummy_type;
  const int scratch_memory_size =
        gemm_dummy_type::ViewTypeAScratch::required_allocation_size() +
        gemm_dummy_type::ViewTypeFScratch::required_allocation_size() +
        gemm_dummy_type::ViewTypeMScratch::required_allocation_size();
  const int scratch_level = scratch_memory_size < 24000 ? 0 : 1;

  // Figure out Team Sizes
  int team_size = 1;
  #if defined(KOKKOS_ENABLE_CUDA)
  if(std::is_same<typename MViewType::execution_space, Kokkos::Cuda>::value)
    team_size = blockA0;
  #endif
  #if defined(KOKKOS_ENABLE_ROCM)
  if(std::is_same<typename MViewType::execution_space, Kokkos::ROCm>::value)
    team_size = blockA0;
  #endif

  KokkosDNN::Impl::CONV2DImpl<typename MViewType::execution_space,
                              AViewType, FViewType, MViewType,
                              blockA0, blockA1, blockB1, 0, 0> conv2d(A, F, stride, M);
  conv2d.run(team_size, vector_length, scratch_level);

/*
  // Call the correct kernel
  if((transA[0]=='N' || transA[0]=='n') && (transB[0]=='N' || transB[0]=='n')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,0,0> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='T' || transA[0]=='t') && (transB[0]=='N' || transB[0]=='n')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,1,0> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='C' || transA[0]=='c') && (transB[0]=='N' || transB[0]=='n')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,2,0> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='N' || transA[0]=='n') && (transB[0]=='T' || transB[0]=='t')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,0,1> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='T' || transA[0]=='t') && (transB[0]=='T' || transB[0]=='t')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,1,1> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='C' || transA[0]=='c') && (transB[0]=='T' || transB[0]=='t')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,2,1> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='N' || transA[0]=='n') && (transB[0]=='C' || transB[0]=='c')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,0,2> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='T' || transA[0]=='t') && (transB[0]=='C' || transB[0]=='c')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,1,2> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
  if((transA[0]=='C' || transA[0]=='c') && (transB[0]=='C' || transB[0]=='c')) {
    KokkosBlas::Impl::GEMMImpl<typename CViewType::execution_space,AViewType,BViewType,CViewType,blockA0,blockA1,blockB1,2,2> gemm(alpha,A,B,beta,C);
    gemm.run(team_size,vector_length,scratch_level);
  }
*/

  Kokkos::Profiling::popRegion();
}
#else
;
#endif //!defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY

};

} // namespace Impl
} // namespace KokkosDNN


//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::GEMM.  This is NOT for users!!!
// All the declarations of full specializations go in this header
// file.  We may spread out definitions (see _DEF macro below) across
// one or more .cpp files.
//

#define KOKKOSDNN_CONV2D_ETI_SPEC_DECL_LAYOUTS( SCALAR, LAYOUTA, LAYOUTF, LAYOUTM, EXEC_SPACE, MEM_SPACE ) \
extern template struct CONV2D< \
     Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<SCALAR**, LAYOUTM, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     false, true>;

#define KOKKOSDNN_CONV2D_ETI_SPEC_INST_LAYOUTS( SCALAR, LAYOUTA, LAYOUTF, LAYOUTM, EXEC_SPACE, MEM_SPACE ) \
template struct CONV2D< \
     Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<SCALAR**, LAYOUTM, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >,  \
     false, true>;

#define KOKKOSDNN_CONV2D_ETI_SPEC_DECL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSDNN_CONV2D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE)

#define KOKKOSDNN_CONV2D_ETI_SPEC_INST( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSDNN_CONV2D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV2D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE)

#include<KokkosDNN_conv2d_tpl_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosDNN_conv2d_eti_spec_decl.hpp>

#endif // KOKKOSDNN_CONV2D_SPEC_HPP_
