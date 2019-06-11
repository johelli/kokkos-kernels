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

#ifndef KOKKOS_DNN_CONV2D_IMPL_HPP_
#define KOKKOS_DNN_CONV2D_IMPL_HPP_

#include<Kokkos_Core.hpp>

namespace KokkosDNN {
namespace Impl {

// Choose Iteration Layout for copying data from global memory into scratch
// On CPUs it is more important to have consecutive write,
// On GPUs it is more important to not jump around in global memory, i.e. 
// have coallesced loads
template<class ExecSpace, class LayoutA, class LayoutAScratch>
struct impl_conv2d_choose_copy_layout {
  typedef LayoutAScratch type;
};

#ifdef KOKKOS_ENABLE_CUDA
template<class LayoutA, class LayoutAScratch>
struct impl_conv2d_choose_copy_layout<Kokkos::Cuda, LayoutA, 
                                      LayoutAScratch> {
  typedef LayoutA type;
};
#endif

// DeepCopy matrix block into scratch

//template<class TeamHandle, class ViewTypeScratch, class ViewType, 
//         class Layout, int blockDim_i, int blockDim_j>
//struct impl_deep_copy_matrix_block;

template<class TeamHandle, class ViewTypeScratch, class ViewType, 
         class Layout> //, int blockDim_i, int blockDim_j>
struct impl_deep_copy_matrix_block {
//struct impl_deep_copy_matrix_block<TeamHandle, ViewTypeScratch, ViewType, 
//                                   Layout, 
//                                   blockDim_i, blockDim_j> {
  typedef typename ViewType::non_const_value_type value_type;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;

  KOKKOS_INLINE_FUNCTION
  static void copy(const TeamHandle& team, 
                   const ViewTypeScratch& A_scr, 
                   const ViewType& A, 
                   const int& blockDim_i,
                   const int& blockDim_j,
                   const int& offset_i, 
                   const int& offset_j) {

    if(offset_i + blockDim_i <= A.extent_int(0) && 
       offset_j + blockDim_j <= A.extent_int(1)) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,blockDim_j), 
                          [&] (const int j) {
        const int idx_j = offset_j + j;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,blockDim_i), 
                            [&] (const int i) {
          const int idx_i = offset_i + i;
          A_scr(i,j) = A(idx_i, idx_j);
        });
      });
    } 
    else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,blockDim_j), 
                          [&] (const int j) {
        const int idx_j = offset_j + j;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,blockDim_i), 
                            [&] (const int i) {
          const int idx_i = offset_i + i;
          A_scr(i,j) = idx_i < A.extent_int(0) && 
            idx_j < A.extent_int(1) ? A(idx_i, idx_j) : ATV::zero();
        });
      });
    }
  }
};

template<class TeamHandle, class ViewTypeScratch, class ViewType> //, 
         //int blockDim_i, int blockDim_j>
struct impl_deep_copy_matrix_block<TeamHandle, ViewTypeScratch, ViewType, 
                                   Kokkos::LayoutRight> { //, 
                                   // blockDim_i, blockDim_j> {
  typedef typename ViewType::non_const_value_type value_type;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;

  KOKKOS_INLINE_FUNCTION
  static void copy(const TeamHandle& team, 
                   const ViewTypeScratch& A_scr, 
                   const ViewType& A,
                   const int& blockDim_i,
                   const int& blockDim_j, 
                   const int& offset_i, 
                   const int& offset_j) {

    if(offset_i + blockDim_i <= A.extent_int(0) && 
       offset_j + blockDim_j <= A.extent_int(1)) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,blockDim_i), 
                          [&] (const int i) {
        const int idx_i = offset_i + i;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,blockDim_j), 
                            [&] (const int j) {
          const int idx_j = offset_j + j;
          A_scr(i,j) = A(idx_i, idx_j);
        });
      });
    } 
    else {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,blockDim_i), 
                          [&] (const int i) {
        const int idx_i = offset_i + i;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,blockDim_j), 
                            [&] (const int j) {
          const int idx_j = offset_j + j;
          A_scr(i,j) = idx_i < A.extent_int(0) && 
            idx_j < A.extent_int(1) ? A(idx_i, idx_j) : ATV::zero();
        });
      });
    }
  }
};

// Update matrix block with new values
template<class TeamHandle, class ViewType, class ViewTypeScratch, 
         class Layout, int blockDim_i, int blockDim_j>
struct impl_update_matrix_block {
  typedef typename ViewType::non_const_value_type value_type;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;

  KOKKOS_INLINE_FUNCTION
  static void update(const TeamHandle& team, const ViewType& C, 
                     const ViewTypeScratch& C_scr,
                     const int& offset_i, const int& offset_j) {

    if(offset_i + blockDim_i <= C.extent_int(0) && 
       offset_j + blockDim_j <= C.extent_int(1)) {
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, blockDim_j), 
                          [&] (const int j) {
        const int idx_j = offset_j + j;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, blockDim_i), 
                            [&] (const int i) {
          const int idx_i = offset_i + i;
          
          C(idx_i, idx_j) = C_scr(i,j);
        });
      });
    } 
    else {
      const int range_i = offset_i + blockDim_i <= C.extent_int(0) ? 
        blockDim_i : C.extent_int(0) % blockDim_i;
      const int range_j = offset_j + blockDim_j <= C.extent_int(1) ? 
        blockDim_j : C.extent_int(1) % blockDim_j;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, range_j), 
                          [&] (const int j) {
        const int idx_j = offset_j + j;


        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, range_i), 
                            [&] (const int i) {
          const int idx_i = offset_i + i;
          
          C(idx_i, idx_j) = C_scr(i,j);
        });
      });
    }
  }
};

template<class TeamHandle, class ViewType, class ViewTypeScratch, 
         int blockDim_i, int blockDim_j>
struct impl_update_matrix_block<TeamHandle, ViewType, ViewTypeScratch,
                                Kokkos::LayoutRight,
                                blockDim_i, blockDim_j> {
  typedef typename ViewType::non_const_value_type value_type;
  typedef Kokkos::Details::ArithTraits<value_type>     ATV;

  KOKKOS_INLINE_FUNCTION
  static void update(const TeamHandle& team, const ViewType& C, 
                     const ViewTypeScratch& C_scr,
                     const int& offset_i, const int& offset_j) {

    if (offset_i + blockDim_i <= C.extent_int(0) && 
        offset_j + blockDim_j <= C.extent_int(1)) {
      
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,blockDim_i), 
                          [&] (const int i) {
        const int idx_i = offset_i + i;

  
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,blockDim_j), 
                            [&] (const int j) {
          const int idx_j = offset_j + j;
          
          C(idx_i, idx_j) = C_scr(i,j);
        });
      });
    }
     
    else {
      const int range_i = offset_i + blockDim_i <= C.extent_int(0) ? 
        blockDim_i : C.extent_int(0) % blockDim_i;
      const int range_j = offset_j + blockDim_j <= C.extent_int(1) ? 
        blockDim_j : C.extent_int(1) % blockDim_j;

      Kokkos::parallel_for(Kokkos::TeamThreadRange(team,range_i), 
                          [&] (const int i) {
        const int idx_i = offset_i + i;

        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,range_j), 
                            [&] (const int j) {
          const int idx_j = offset_j + j;

          C(idx_i, idx_j) = C_scr(i,j);
        });
      });
    }
  }
};

// Compute a single C block 8 B block, also do an in-place no-additional 
// blocking team conv2d
template<class TeamHandle, class ViewTypeA, class ViewTypeF, 
         class ViewTypeC>
KOKKOS_INLINE_FUNCTION
void impl_team_conv2d_block(const TeamHandle& team, 
                            const ViewTypeC& C, 
                            const ViewTypeA& A, 
                            const ViewTypeF& F, 
                            const int stride) {
  typedef typename ViewTypeC::non_const_value_type ScalarC;

// GNU COMPILER BUG WORKAROUND
#if defined(KOKKOS_COMPILER_GNU) || !defined(__CUDA_ARCH__)
  int blockA0 = A.extent_int(0);
  int blockA1 = A.extent_int(1);
  int blockF0 = F.extent_int(0);
  int blockF1 = F.extent_int(1);
  int blockC0 = C.extent_int(0);
  int blockC1 = C.extent_int(1);

#else
  const int blockA0 = A.extent_int(0);
  const int blockA1 = A.extent_int(1);
  const int blockF0 = F.extent_int(0);
  const int blockF1 = F.extent_int(1);
  const int blockC0 = C.extent_int(0);
  const int blockC1 = C.extent_int(1);
#endif

  Kokkos::parallel_for(Kokkos::TeamThreadRange(team, blockC0), 
                      [&] (const int C_i) {
#if defined(__CUDA_ARCH__) || !defined(KOKKOS_ENABLE_OPENMP)
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(team, blockC1), 
                        [&] (const int C_j) {
#else
  #if defined(KOKKOS_COMPILER_GNU)
    #if (KOKKOS_COMPILER_GNU > 485 )
    #pragma omp simd
    #endif
  #else
    #pragma omp simd
  #endif

    for (int C_j = 0; C_j < blockC1; ++C_j) {
#endif
      ScalarC C_ij = 0.0;

      int A_i_offset = stride * C_i;
      int A_j_offset = stride * C_j; 

      for (int F_i = 0; F_i < blockF0; ++F_i) {
        for (int F_j = 0; F_j < blockF1; ++F_j) {
          C_ij += 
            A(A_i_offset + F_i, A_j_offset + F_j) * F(F_i, F_j);



          std::cout << "\n(i,j): " << C_i << " " << C_j << " C_ij: " << C_ij
            << " A_i_offset: " << A_i_offset << " A_j_offset: " << A_j_offset
            << " F_i: " << F_i << " F_j: " << F_j << " BlockF0: " << blockF0 
            << " BlockF1: " << blockF1 << " A(of + F_i, of + F_j): " 
            << A(A_i_offset + F_i, A_j_offset + F_j) << " F(F_i, F_j): "
            << F(F_i, F_j) << " blockA0: " << blockA0 << " blockA1: "
            << blockA1 << " blockC0: " << blockC0 << " blockC1: " << blockC1 << std::endl;


        } 
      }

      C(C_i, C_j) += C_ij;
#if defined(__CUDA_ARCH__) || !defined(KOKKOS_ENABLE_OPENMP)
    });
#else
    }
#endif
  });
}

template<class ExecSpace, class ViewTypeA, class ViewTypeF, 
         class ViewTypeC, 
 //        int blockA0, int blockA1, int blockF0, int blockF1, 
         int blockC0, int blockC1>
struct CONV2DImpl {
  ViewTypeA A;
  ViewTypeF F;
  ViewTypeC C;
  typedef typename ViewTypeA::non_const_value_type ScalarA;
  typedef typename ViewTypeF::non_const_value_type ScalarF;
  typedef typename ViewTypeC::non_const_value_type ScalarC;
  typedef typename Kokkos::TeamPolicy<ExecSpace>   PolicyType;
  typedef typename PolicyType::member_type         MemberType;

  const int num_blocks_0;
  const int num_blocks_1;
  int blockA0, blockA1;
  int blockF0, blockF1;
  int scratch_level;

  const int stride;

/*
  typedef Kokkos::View<ScalarA[blockA0][blockA1], Kokkos::LayoutLeft, 
                       typename ExecSpace::scratch_memory_space>
    ViewTypeAScratch;
  typedef Kokkos::View<ScalarF[blockF0][blockF1], Kokkos::LayoutRight, 
                       typename ExecSpace::scratch_memory_space>
    ViewTypeFScratch;
*/

  typedef Kokkos::View<ScalarC[blockC0][blockC1], Kokkos::LayoutRight, 
                       typename ExecSpace::scratch_memory_space>
    ViewTypeCScratch;

  CONV2DImpl(const ViewTypeA& A_, const ViewTypeF& F_, 
             const int stride_, const ViewTypeC& C_) : 
    A(A_),
    F(F_),
    C(C_),
//    num_blocks_0((C.extent_int(0) + blockA0 - 1) / blockA0),
//    num_blocks_1((C.extent_int(1) + blockF1 - 1) / blockF1) {
    num_blocks_0((C.extent_int(0) + blockC0 - 1) / blockC0),
    num_blocks_1((C.extent_int(1) + blockC1 - 1) / blockC1),
    stride(stride_),
    blockF0(F_.extent_int(0)),
    blockF1(F_.extent_int(1)) { 

    scratch_level = 0;
    blockA0 = blockF0 + (blockC0 - 1) * stride;
    blockA1 = blockF1 + (blockC1 - 1) * stride;

  }

  void run(int team_size, int vector_length) { //, int scr_level) {

//    scratch_level = scr_level;

    typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, 
                       typename ExecSpace::scratch_memory_space>
      ViewTypeAScratch;
    typedef Kokkos::View<ScalarF**, Kokkos::LayoutRight, 
                       typename ExecSpace::scratch_memory_space>
      ViewTypeFScratch;

    size_t viewA_shared_size = ViewTypeAScratch::shmem_size(team_size, blockA0 * blockA1);
    size_t viewF_shared_size = ViewTypeFScratch::shmem_size(team_size, blockF0 * blockF1);

    int scratch_memory_size =
      viewA_shared_size + viewF_shared_size + 
      ViewTypeCScratch::required_allocation_size();

    scratch_level = scratch_memory_size < 24000 ? 0 : 1;

    // Launch bounds???
    Kokkos::TeamPolicy<ExecSpace,Kokkos::LaunchBounds<384,2>> policy(
        num_blocks_0 * num_blocks_1, team_size, vector_length);

    Kokkos::parallel_for("KokkosDNN::conv2d", 
                         policy.set_scratch_size(scratch_level, 
                           Kokkos::PerTeam(scratch_memory_size)), *this);
  }

  KOKKOS_INLINE_FUNCTION
  void operator() (const typename Kokkos::TeamPolicy<ExecSpace>::
                   member_type& team) const {
    typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, 
                       typename ExecSpace::scratch_memory_space>
      ViewTypeAScratch;
    typedef Kokkos::View<ScalarF**, Kokkos::LayoutRight, 
                       typename ExecSpace::scratch_memory_space>
      ViewTypeFScratch;
 
    // This team is responsible for computing a single block of C
    const int league_rank = team.league_rank();
//    const int num_blocks = num_blocks_1;
    const int C_i_offset = (league_rank / num_blocks_1) * blockC0;
    const int C_j_offset = (league_rank % num_blocks_1) * blockC1;

//    ViewTypeAScratch A_scr(team.team_scratch(scratch_level));
//    ViewTypeFScratch F_scr(team.team_scratch(scratch_level));

    ViewTypeAScratch A_scr(team.team_scratch(scratch_level), blockA0, blockA1);
    ViewTypeFScratch F_scr(team.team_scratch(scratch_level), blockF0, blockF1);

    ViewTypeCScratch C_scr(team.team_scratch(scratch_level));

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team,blockC0), 
                        [&] (const int i) {
      Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,blockC1), 
                          [&] (const int j) {
        C_scr(i,j) = 0;
      });
    });
    team.team_barrier();

/*
    // Load F block (the whole filter) into scratch
    KokkosDNN::Impl::impl_deep_copy_matrix_block<MemberType,
                                ViewTypeFScratch, 
                                ViewTypeF,
                                typename impl_conv2d_choose_copy_layout<
                                      ExecSpace,
                                    typename ViewTypeF::array_layout,
                                    typename ViewTypeFScratch::array_layout>::type,
                                blockF0, 
                                blockF1>::copy(team, F_scr, F, 0, 0);
*/

    // Move along the inner dimension in blocks
//    const int length = C.extent_int(1);  
//    for(int C_j = 0; C_j < length; C_j += blockC1) {

      // offsets for A block deep copy
//      int A_i_offset = stride * blockC0 * C_i_offset;
//      int A_j_offset = stride * blockC1 * (C_j / blockC1); 

    int A_i_offset = (league_rank / num_blocks_1) * blockC0 * stride; 
    int A_j_offset = (league_rank % num_blocks_1) * blockC1 * stride;
    
    // Load A block into scratch
    KokkosDNN::Impl::impl_deep_copy_matrix_block<MemberType,
                                ViewTypeAScratch, 
                                ViewTypeA,
                                typename impl_conv2d_choose_copy_layout<
                                    ExecSpace,
                                    typename ViewTypeA::array_layout,
                                    typename ViewTypeAScratch::array_layout>::type>::copy( //,
                                // blockA0, 
                                // blockA1>::copy(
                                  team, 
                                  A_scr, 
                                  A,
                                  blockA0,
                                  blockA1, 
                                  A_i_offset, 
                                  A_j_offset);

    // Load F block (the whole filter) into scratch
    KokkosDNN::Impl::impl_deep_copy_matrix_block<MemberType,
                                ViewTypeFScratch, 
                                ViewTypeF,
                                typename impl_conv2d_choose_copy_layout<
                                      ExecSpace,
                                    typename ViewTypeF::array_layout,
                                    typename ViewTypeFScratch::array_layout>::type>::copy( //,
                                //blockF0, 
                                //blockF1>::copy(
                                  team, 
                                  F_scr, 
                                  F, 
                                  blockF0,
                                  blockF1,
                                  0, 0);

/*
    // Load F block (the whole filter) into scratch
    KokkosDNN::Impl::impl_deep_copy_matrix_block<MemberType,
                                ViewTypeFScratch, 
                                ViewTypeF,
                                typename impl_conv2d_choose_copy_layout<
                                      ExecSpace,
                                    typename ViewTypeF::array_layout,
                                    typename ViewTypeFScratch::array_layout>::type,
                                blockF0, 
                                blockF1>::copy(team, F_scr,
                                               F, 0, 0);
*/

    // Wait for A and F block to be in scratch memory
    team.team_barrier();

    // Add contribution from convolving A with filter F and output C block
    impl_team_conv2d_block(team, C_scr, A_scr, F_scr, stride);

    for (int i = 0; i < blockC0; ++i) {
      for (int j = 0; j < blockC1; ++j) {
        std::cout << "\n(i,j): " << i << " " << j 
          << " C_ij: " << C_scr(i, j) 
          << " league rank: " << league_rank 
          << " C_i_offset: " << C_i_offset 
          << " C_j_offset: " << C_j_offset << std::endl;
      }
    }


    // Wait for subblock computation to be done before updating  
    team.team_barrier();

//    }

    // Write back the C block from scratch to main memory
    KokkosDNN::Impl::impl_update_matrix_block<MemberType,
                             ViewTypeC, 
                             ViewTypeCScratch,
                             typename ViewTypeC::array_layout,
                             blockC0, 
                             blockC1>::update(team, 
                                              C, 
                                              C_scr, 
                                              C_i_offset, 
                                              C_j_offset);
  }
};

} // namespace Impl

} // namespace KokkosDNN

#endif // KOKKOS_DNN_CONV2D_HPP_
