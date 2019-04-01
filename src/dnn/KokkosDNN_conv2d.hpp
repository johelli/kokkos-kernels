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
#ifndef KOKKOSDNN_CONV2D_HPP_
#define KOKKOSDNN_CONV2D_HPP_

/// \file KokkosDNN_conv2d.hpp

#include <KokkosKernels_Macros.hpp>
#include <KokkosDNN_conv2d_spec.hpp>
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits>

namespace KokkosDNN {

/// \brief 2D direct convolution: M_ik = SUM_c=1^C A_ic * F_kc,
/// where * is the correlation operator, i is image receptive 
/// field, k is the filter, c is the channel #.
///
/// \tparam AViewType Input image matrix, as a 2-D Kokkos::View
/// \tparam FViewType Input filter matrix, as a 2-D Kokkos::View
/// \tparam MViewType Output matrix, as a nonconst 2-D Kokkos::View
///
/// \param A [in] Input matrix, as a 2-D Kokkos::View
/// \param F [in] Input matrix, as a 2-D Kokkos::View
/// \param M [in/out] Output vector, as a nonconst 2-D Kokkos::View
/// \param bias [in] Input bias matrix for affine shift, as a 2-D 
///   Kokkos::View
/// \param stride [in] Input integer, filter movement over image
///   default = 1 
template<class AViewType,
         class FViewType,
         class CViewType>
void conv2d(const AViewType& A,
            const FViewType& F,
            const int stride,
            const CViewType& C)          
{

  #if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert (Kokkos::Impl::is_view<AViewType>::value,
                 "AViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<FViewType>::value,
                 "FViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<CViewType>::value,
                 "CViewType must be a Kokkos::View.");
  static_assert (static_cast<int> (AViewType::rank) == 2,
                 "AViewType must have rank 2.");
  static_assert (static_cast<int> (FViewType::rank) == 2,
                 "FViewType must have rank 2.");
  static_assert (static_cast<int> (CViewType::rank) == 2,
                 "CViewType must have rank 2.");

  // Check validity of transpose argument
/*
  bool valid_transA = (transA[0] == 'N') || (transA[0] == 'n') ||
                      (transA[0] == 'T') || (transA[0] == 't') ||
                      (transA[0] == 'C') || (transA[0] == 'c');
  bool valid_transB = (transB[0] == 'N') || (transB[0] == 'n') ||
                      (transB[0] == 'T') || (transB[0] == 't') ||
                      (transB[0] == 'C') || (transB[0] == 'c');
  if(!(valid_transA && valid_transB)) {
    std::ostringstream os;
    os << "KokkosBlas::gemm: transA[0] = '" << transA[0] 
      << " transB[0] = '" << transB[0] << "'. " 
      << "Valid values include 'N' or 'n' (No transpose), 'T' or 't' " 
      "(Transpose), and 'C' or 'c' (Conjugate transpose).";
    Kokkos::Impl::throw_runtime_exception (os.str ());
  }

  // Check compatibility of dimensions at run time.
  bool A_t = !(transA[0] == 'N' || transA[0] == 'n');
  bool B_t = !(transB[0] == 'N' || transB[0] == 'n');
  int64_t A0 = A.extent(0);
  int64_t A1 = A.extent(1);
  int64_t B0 = B.extent(0);
  int64_t B1 = B.extent(1);
  int64_t C0 = C.extent(0);
  int64_t C1 = C.extent(1);

  if ( ((A_t?A1:A0) != C0) ||
       ((B_t?B0:B1) != C1) ||
       ((A_t?A0:A1) != (B_t?B1:B0)) ) {
      std::ostringstream os;
      os << "KokkosBlas::gemm: Dimensions of A, B, and C do not match: "
         << "transA: " << transA[0] << " transB: " << transB[0]
         << " A: " << A.extent(0) << " x " << A.extent(1)
         << " B: " << B.extent(0) << " x " << B.extent(1)
         << " C: " << C.extent(0) << " x " << C.extent(1);
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }
  #endif // KOKKOSKERNELS_DEBUG_LEVEL > 0

  // Return if degenerated matrices are provided
  if((A.extent(0) == 0) || (A.extent(1) == 0) || (C.extent(1) == 0))
    return;
*/

  // Minimize the number of Impl::CONV2D instantiations, by
  // standardizing on particular View specializations for its template
  // parameters.
  typedef Kokkos::View<typename AViewType::const_value_type**,
    typename AViewType::array_layout,
    typename AViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;

  typedef Kokkos::View<typename FViewType::const_value_type**,
    typename FViewType::array_layout,
    typename FViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > FVT;

  typedef Kokkos::View<typename CViewType::non_const_value_type**,
    typename CViewType::array_layout,
    typename CViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > CVT;

  typedef Impl::CONV2D<AVT, FVT, CVT> impl_type;
  impl_type::conv2d (A, F, stride, C);
}

} // namespace KokkosDNN

#endif // KOKKOSDNN_CONV2D_HPP_
