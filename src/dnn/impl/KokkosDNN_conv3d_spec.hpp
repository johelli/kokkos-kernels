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
#ifndef KOKKOSDNN_CONV3D_SPEC_HPP_
#define KOKKOSDNN_CONV3D_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include<KokkosDNN_conv3d_impl.hpp>
#endif

namespace KokkosDNN {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template<class AVT, class FVT, class CVT>
struct conv3d_eti_spec_avail {
  enum : bool { value = false };
};
}
}


//
// Macro for declaration of full specialization availability
// KokkosDNN::Impl::CONV3D.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSDNN_CONV3D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, LAYOUTA, LAYOUTF, LAYOUTC, EXEC_SPACE, MEM_SPACE ) \
    template<> \
    struct conv3d_eti_spec_avail< \
         Kokkos::View<const SCALAR***, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<const SCALAR***, LAYOUTF, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR***, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
         > { enum : bool { value = true }; };

#define KOKKOSDNN_CONV3D_ETI_SPEC_AVAIL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSDNN_CONV3D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_AVAIL_LAYOUT( SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE)

// Include the actual specialization declarations
#include<KokkosDNN_conv3d_tpl_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosDNN_conv3d_eti_spec_avail.hpp>

namespace KokkosDNN {
namespace Impl {

//
// conv3d
//

// Implementation of KokkosDNN::conv3d.
template<class AViewType,
         class FViewType,
         class CViewType,
         bool tpl_spec_avail = conv3d_tpl_spec_avail<AViewType, FViewType, CViewType>::value,
         bool eti_spec_avail = conv3d_eti_spec_avail<AViewType, FViewType, CViewType>::value
         >
struct CONV3D {
  static void conv3d(const AViewType& A,
                     const FViewType& F,
                     const int stride, 
                     const CViewType& C)
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
{
  static_assert (Kokkos::Impl::is_view<AViewType>::value,
                 "AViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<FViewType>::value,
                 "FViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<CViewType>::value,
                 "CViewType must be a Kokkos::View.");
  static_assert (static_cast<int> (AViewType::rank) == 3,
                 "AViewType must have rank 3.");
  static_assert (static_cast<int> (FViewType::rank) == 3,
                 "FViewType must have rank 3.");
  static_assert (static_cast<int> (CViewType::rank) == 3,
                 "CViewType must have rank 3.");

  Kokkos::Profiling::pushRegion(eti_spec_avail?"KokkosDNN::conv3d[ETI]":"KokkosDNN::conv3d[noETI]");
  // Figure out Scalar Types
  typedef typename AViewType::non_const_value_type ScalarA;
  typedef typename FViewType::non_const_value_type ScalarF;
  typedef typename CViewType::non_const_value_type ScalarC;

  // Always use full Filter
//  const int blockF0 = F.extent_int(0);
//  const int blockF1 = F.extent_int(1);
   
//  static constexpr int num_strides = 4;
//  int num_strides = 200 / (blockF0 * blockF1) < 1 ? 1 : 200 / (blockF0 * blockF1);
  
  // Define Blocking sizes (this will be used for scratch spaces)
  // Creates A blocks that are strided perfectly given the filter and
  // stride length
//  const int blockA0 = blockF0 + num_strides * stride; 
//  const int blockA1 = blockF1 + num_strides * stride;

  // Always use full Filter
//  static constexpr int blockA0 = 7;
//  static constexpr int blockA1 = 7;

//  static constexpr int blockF0 = 3;
//  static constexpr int blockF1 = 3;
   
  static constexpr int blockC0 = 5;
  static constexpr int blockC1 = 5;
  static constexpr int blockC2 = 5;

/* 
  static constexpr int blockA1 = 
    (sizeof(ScalarA) * blockA0 * 16 + 
     sizeof(ScalarF) * 16 * blockF1 + 
     sizeof(ScalarC) * blockA0 * blockF1 < 24000) ? 
        16 : (sizeof(ScalarA) * blockA0 * 8 + 
              sizeof(ScalarF) * 8 * blockF1 + 
              sizeof(ScalarC) * blockA0 * blockF1 < 24000) ? 
          8 :  (sizeof(ScalarA) * blockA0 * 4 + 
                sizeof(ScalarF) * 4 * blockF1 + 
                sizeof(ScalarC) * blockA0 * blockF1 < 24000) ? 
            4 : 16;
*/


/*
  static constexpr int blockC0 = 24;
  static constexpr int blockC1 = 24;
*/

  // C block size dependent on A, F block dims
//  const int blockC0 = (blockA0 - blockF0) / stride + 1;
//  const int blockC1 = (blockA1 - blockF1) / stride + 1;
 
  //??? 
  const int vector_length = 32;

  // Compute scratch space size
//  typedef KokkosDNN::Impl::CONV3DImpl<typename CViewType::execution_space, 
//                                       AViewType, FViewType, CViewType, 
  //                                     blockA0, blockA1, blockF0, blockF1,
//                                       blockC0, blockC1> conv3d_dummy_type;
  //const int scratch_memory_size =
  //      conv3d_dummy_type::ViewTypeAScratch::required_allocation_size() +
  //      conv3d_dummy_type::ViewTypeFScratch::required_allocation_size() +
  //      conv3d_dummy_type::ViewTypeCScratch::required_allocation_size();
  // const int scratch_level = scratch_memory_size < 24000 ? 0 : 1;

  // Figure out Team Sizes
  int team_size = 1;
/*  #if defined(KOKKOS_ENABLE_CUDA)
  if(std::is_same<typename CViewType::execution_space, Kokkos::Cuda>::value)
    team_size = blockC0;
  #endif
  #if defined(KOKKOS_ENABLE_ROCM)
  if(std::is_same<typename CViewType::execution_space, Kokkos::ROCm>::value)
    team_size = blockC0;
  #endif
*/

  KokkosDNN::Impl::CONV3DImpl<typename CViewType::execution_space,
                              AViewType, FViewType, CViewType,
  //                            blockA0, blockA1, blockF0, blockF1,
                              blockC0, blockC1, blockC2> conv3d(A, F, stride, C);
  conv3d.run(team_size, vector_length);  //, scratch_level);

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
// KokkosDNN::Impl::CONV3D.  This is NOT for users!!!
// All the declarations of full specializations go in this header
// file.  We may spread out definitions (see _DEF macro below) across
// one or more .cpp files.
//

#define KOKKOSDNN_CONV3D_ETI_SPEC_DECL_LAYOUTS( SCALAR, LAYOUTA, LAYOUTF, LAYOUTC, EXEC_SPACE, MEM_SPACE ) \
extern template struct CONV3D< \
     Kokkos::View<const SCALAR***, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR***, LAYOUTF, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<SCALAR***, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     false, true>;

#define KOKKOSDNN_CONV3D_ETI_SPEC_INST_LAYOUTS( SCALAR, LAYOUTA, LAYOUTF, LAYOUTC, EXEC_SPACE, MEM_SPACE ) \
template struct CONV3D< \
     Kokkos::View<const SCALAR***, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR***, LAYOUTF, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<SCALAR***, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >,  \
     false, true>;

#define KOKKOSDNN_CONV3D_ETI_SPEC_DECL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSDNN_CONV3D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE)

#define KOKKOSDNN_CONV3D_ETI_SPEC_INST( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSDNN_CONV3D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutLeft, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSDNN_CONV3D_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutRight, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE)

#include<KokkosDNN_conv3d_tpl_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosDNN_conv3d_eti_spec_decl.hpp>

#endif // KOKKOSDNN_CONV3D_SPEC_HPP_
