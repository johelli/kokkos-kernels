/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
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
#ifndef KOKKOSDNN_CONV3D_HPP_
#define KOKKOSDNN_CONV3D_HPP_

/// \file KokkosDNN_conv3d.hpp

#include <KokkosKernels_Macros.hpp>
#include <KokkosDNN_conv3d_spec.hpp>
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosDNN {

/// \brief 3D direct convolution: C =  A * F,
/// where * is the correlation operator. 
///
/// \tparam AViewType Input image matrix, as a 3D Kokkos::View
/// \tparam FViewType Input filter matrix, as a 3D Kokkos::View
/// \tparam CViewType Output matrix, as a nonconst 3D Kokkos::View
///
/// \param A [in] Input matrix, as a 3D Kokkos::View
/// \param F [in] Input matrix, as a 3D Kokkos::View
/// \param stride [in] Input integer, filter movement over image
///   default = 1 
/// \param C [in/out] Output vector, as a nonconst 3D Kokkos::View
template<class AViewType,
         class FViewType,
         class CViewType>
void conv3d(const AViewType& A,
            const FViewType& F,
            const int stride,
            const CViewType& C)          
{
  // Return if degenerated matrices are provided
  if((A.extent(0) == 0) || (A.extent(1) == 0) || (A.extent(2) == 0) || 
     (F.extent(0) == 0) || (F.extent(1) == 0) || (F.extent(2) == 0) ||
     (C.extent(0) == 0) || (C.extent(1) == 0) || (C.extent(2) == 0))
    return;

  #if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
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

  // Check compatibility of dimensions at run time.
  int64_t A0 = A.extent(0);
  int64_t A1 = A.extent(1);
  int64_t A2 = A.extent(2);
  int64_t F0 = F.extent(0);
  int64_t F1 = F.extent(1);
  int64_t F2 = F.extent(2);
  int64_t C0 = C.extent(0);
  int64_t C1 = C.extent(1);
  int64_t C2 = C.extent(2);

  // Required dimensions of C, given A and F
  int64_t M = (A0 - F0) / stride + 1;
  int64_t N = (A1 - F1) / stride + 1; 
  int64_t O = (A2 - F2) / stride + 1; 

  if ((F0 % 2 == 0) || (F1 % 2 == 0) || (F2 % 2 == 0)) {
    std::ostringstream os;
      os << "KokkosDNN::conv3d: Dimensions of filter F must be odd: "
         << " F: " << F0 << " x " << F1 << " x " << F2 ;
      Kokkos::Impl::throw_runtime_exception(os.str());
  }
  else if (C0 != M || C1 != N || C2 != O) {
      std::ostringstream os;
      os << "KokkosDNN::conv3d: Dimensions of A, F, and C do not match: "
         << " A: " << A0 << " x " << A1 << " x " << A2
         << " F: " << F0 << " x " << F1 << " x " << F2
         << " C: " << C0 << " x " << C1 << " x " << C2
         << " Required C: " << M << " x " << N << " x " << O;
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }
  #endif // KOKKOSKERNELS_DEBUG_LEVEL > 0

  // Minimize the number of Impl::CONV3D instantiations, by
  // standardizing on particular View specializations for its template
  // parameters.
  typedef Kokkos::View<typename AViewType::const_value_type***,
    typename AViewType::array_layout,
    typename AViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;

  typedef Kokkos::View<typename FViewType::const_value_type***,
    typename FViewType::array_layout,
    typename FViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > FVT;

  typedef Kokkos::View<typename CViewType::non_const_value_type***,
    typename CViewType::array_layout,
    typename CViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > CVT;

  typedef Impl::CONV3D<AVT, FVT, CVT> impl_type;
  
  impl_type::conv3d(A, F, stride, C);
}

} // namespace KokkosDNN

#endif // KOKKOSDNN_CONV3D_HPP_
