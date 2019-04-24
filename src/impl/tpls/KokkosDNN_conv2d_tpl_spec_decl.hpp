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
// PROCUREMENT OF SUFSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSDNN_CONV2D_TPL_SPEC_DECL_HPP_
#define KOKKOSDNN_CONV2D_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
extern "C" void dconv2d_(const int* M, const int* N, const int* K,
                         const double* A, const int* LDA,
                         const double* F, const int* LDF,
                         const int* stride,
                         double* C, const int* LDC);
extern "C" void sconv2d_(const int* M, const int* N, const int* K,
                         const float* A, const int* LDA,
                         const float* F, const int* LDF,
                         const int* stride, 
                         float* C, const int* LDC);
extern "C" void zconv2d_(const int* M, const int* N, const int* K,
                         const std::complex<double>* A, const int* LDA,
                         const std::complex<double>* F, const int* LDF,
                         const int* stride, 
                         std::complex<double>* C, const int* LDC);
extern "C" void cconv2d_(const int* M, const int* N, const int* K,
                         const std::complex<float>* A, const int* LDA,
                         const std::complex<float>* F, const int* LDF,
                         const int* stride, 
                         std::complex<float>* C, const int* LDC);

namespace KokkosDNN {
namespace Impl {

#define KOKKOSDNN_DCONV2D_BLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const double**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef double SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
 \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,double]"); \
    const int M = C.extent(0); \
    const int N = C.extent(1); \
    const int K = A.extent(1); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) { \
      dconv2d_(&M, &N, &K, \
               A.data(), &LDA, \
               F.data(), &LDF, \
               &stride, \
               C.data(), &LDC); \
    } \
    if(A_is_lr && F_is_lr && C_is_lr ) { \
      dconv2d_(&N, &M, &K, \
               F.data(), &LDF, \
               A.data(), &LDA, \
               &stride, \
               C.data(), &LDC); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSDNN_SCONV2D_BLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const float**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef float SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
      \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,float]"); \
    const int M = C.extent(0); \
    const int N = C.extent(1); \
    const int K = A.extent(1); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) { \
      sconv2d_(&M, &N, &K, \
               A.data(), &LDA, \
               F.data(), &LDF, \
               &stride, \
               C.data(), &LDC); \
    } \
    if(A_is_lr && F_is_lr && C_is_lr ) { \
      sconv2d_(&N, &M, &K, \
               F.data(), &LDF, \
               A.data(), &LDA, \
               &stride, \
               C.data(), &LDC); \
    } \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSDNN_ZCONV2D_BLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const Kokkos::complex<double>**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<double>**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<double> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
      \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,complex<double>]"); \
    const int M = C.extent(0); \
    const int N = C.extent(1); \
    const int K = A.extent(1); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) \
      zconv2d_(&M, &N, &K, \
               reinterpret_cast<const std::complex<double>*>(A.data()), \
               &LDA, \
               reinterpret_cast<const std::complex<double>*>(F.data()), \
               &LDF, \
               &stride, \
               reinterpret_cast<std::complex<double>*>(C.data()), \
               &LDC); \
    if(A_is_lr && F_is_lr && C_is_lr ) \
      zconv2d_(&N, &M, &K, \
               reinterpret_cast<const std::complex<double>*>(F.data()), \
               &LDF, \
               reinterpret_cast<const std::complex<double>*>(A.data()), \
               &LDA, \
               &stride, \
               reinterpret_cast<std::complex<double>*>(C.data()), \
               &LDC); \
    Kokkos::Profiling::popRegion(); \
  } \
}; \

#define KOKKOSDNN_CCONV2D_BLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const Kokkos::complex<float>**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<float>**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<float> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
      \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,complex<float>]"); \
    const int M = C.extent(0); \
    const int N = C.extent(1); \
    const int K = A.extent(1); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) \
      cconv2d_(&M, &N, &K, \
               reinterpret_cast<const std::complex<float>*>(A.data()), \
               &LDA, \
               reinterpret_cast<const std::complex<float>*>(F.data()), \
               &LDF, \
               &stride, \
               reinterpret_cast<std::complex<float>*>(C.data()), \
               &LDC); \
    if(A_is_lr && F_is_lr && C_is_lr ) \
      cconv2d_(&N, &M, &K, \
               reinterpret_cast<const std::complex<float>*>(F.data()), \
               &LDF, \
               reinterpret_cast<const std::complex<float>*>(A.data()), \
               &LDA, \
               &stride, \
               reinterpret_cast<std::complex<float>*>(C.data()), \
               &LDC); \
    Kokkos::Profiling::popRegion(); \
  } \
};

KOKKOSDNN_DCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSDNN_DCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSDNN_DCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSDNN_DCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSDNN_SCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSDNN_SCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSDNN_SCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSDNN_SCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSDNN_ZCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSDNN_ZCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSDNN_ZCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSDNN_ZCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSDNN_CCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSDNN_CCONV2D_BLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSDNN_CCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSDNN_CCONV2D_BLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

}
}
#endif // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include<KokkosDNN_tpl_spec.hpp>

namespace KokkosDNN {
namespace Impl {

#define KOKKOSDNN_DCONV2D_CUBLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const double**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const double**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef double SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
 \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,double]"); \
    const int M = static_cast<int> (C.extent(0)); \
    const int N = static_cast<int> (C.extent(1)); \
    const int K = static_cast<int> (A.extent(1)); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    KokkosDNN::Impl::CudaBlasSingleton & s = KokkosDNN::Impl::CudaBlasSingleton::singleton(); \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) \
      cublasDconv2d(s.handle, M, N, K, \
                    A.data(), LDA, \
                    F.data(), LDF, \
                    stride, \
                    C.data(), LDC); \
    if(A_is_lr && F_is_lr && C_is_lr ) \
      cublasDconv2d(s.handle, N, M, K, \
                    F.data(), LDF, \
                    A.data(), LDA, \
                    stride, \
                    C.data(), LDC); \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSDNN_SCONV2D_CUBLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const float**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const float**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef float SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
      \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,float]"); \
    const int M = static_cast<int> (C.extent(0)); \
    const int N = static_cast<int> (C.extent(1)); \
    const int K = static_cast<int> (A.extent(A_t?0:1)); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    KokkosDNN::Impl::CudaBlasSingleton & s = KokkosDNN::Impl::CudaBlasSingleton::singleton(); \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) \
      cublasSconv2d(s.handle, M, N, K, \
                    A.data(), LDA, \
                    F.data(), LDF, \
                    stride, \
                    C.data(), LDC); \
    if(A_is_lr && F_is_lr && C_is_lr ) \
      cublasSconv2d(s.handle, N, M, K, \
                    F.data(), LDF, \
                    A.data(), LDA, \
                    stride, \
                    C.data(), LDC); \
    Kokkos::Profiling::popRegion(); \
  } \
};

#define KOKKOSDNN_ZCONV2D_CUBLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const Kokkos::complex<double>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const Kokkos::complex<double>**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<double>**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<double> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
      \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,complex<double>]"); \
    const int M = static_cast<int> (C.extent(0)); \
    const int N = static_cast<int> (C.extent(1)); \
    const int K = static_cast<int> (A.extent(A_t?0:1)); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    KokkosDNN::Impl::CudaBlasSingleton & s = KokkosDNN::Impl::CudaBlasSingleton::singleton(); \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) \
      cublasZconv2d(s.handle, M, N, K, \
                    reinterpret_cast<const cuDoubleComplex*>(A.data()), \
                    LDA, \
                    reinterpret_cast<const cuDoubleComplex*>(F.data()), \
                    LDF, \
                    stride, \
                    reinterpret_cast<cuDoubleComplex*>(C.data()), \
                    LDC); \
    if(A_is_lr && F_is_lr && C_is_lr ) \
      cublasZconv2d(s.handle, N, M, K, \
                    reinterpret_cast<const cuDoubleComplex*>(F.data()), \
                    LDF, \
                    reinterpret_cast<const cuDoubleComplex*>(A.data()), \
                    LDA, \
                    stride, \
                    reinterpret_cast<cuDoubleComplex*>(C.data()), \
                    LDC); \
    Kokkos::Profiling::popRegion(); \
  } \
}; \

#define KOKKOSDNN_CCONV2D_CUBLAS( LAYOUTA, LAYOUTF, LAYOUTC, MEM_SPACE, ETI_SPEC_AVAIL ) \
template<class ExecSpace> \
struct CONV2D< \
     Kokkos::View<const Kokkos::complex<float>**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const Kokkos::complex<float>**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<Kokkos::complex<float>**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     true, ETI_SPEC_AVAIL> { \
  typedef Kokkos::complex<float> SCALAR; \
  typedef Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > AViewType; \
  typedef Kokkos::View<const SCALAR**, LAYOUTF, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > FViewType; \
  typedef Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<ExecSpace, MEM_SPACE>, \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > CViewType; \
      \
  static void \
  conv2d (const AViewType& A, \
          const FViewType& F, \
          const int stride, \
          const CViewType& C) { \
    \
    Kokkos::Profiling::pushRegion("KokkosDNN::conv2d[TPL_BLAS,complex<float>]"); \
    const int M = static_cast<int> (C.extent(0)); \
    const int N = static_cast<int> (C.extent(1)); \
    const int K = static_cast<int> (A.extent(A_t?0:1)); \
    \
    bool A_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTA>::value; \
    bool F_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTF>::value; \
    bool C_is_lr = std::is_same<Kokkos::LayoutRight,LAYOUTC>::value; \
    \
    const int AST = A_is_lr?A.stride(0):A.stride(1), LDA = AST == 0 ? 1 : AST; \
    const int FST = F_is_lr?F.stride(0):F.stride(1), LDF = FST == 0 ? 1 : FST; \
    const int CST = C_is_lr?C.stride(0):C.stride(1), LDC = CST == 0 ? 1 : CST; \
    \
    KokkosDNN::Impl::CudaBlasSingleton & s = KokkosDNN::Impl::CudaBlasSingleton::singleton(); \
    if(!A_is_lr && !F_is_lr && !C_is_lr ) \
      cublasCconv2d(s.handle, M, N, K, \
                    reinterpret_cast<const cuComplex*>(A.data()), \
                    LDA, \
                    reinterpret_cast<const cuComplex*>(F.data()), \
                    LDF, \
                    stride, \
                    reinterpret_cast<cuComplex*>(C.data()), \
                    LDC); \
    if(A_is_lr && F_is_lr && C_is_lr ) \
      cublasCconv2d(s.handle, N, M, K, \
                    reinterpret_cast<const cuComplex*>(F.data()), \
                    LDF, \
                    reinterpret_cast<const cuComplex*>(A.data()), \
                    LDA, \
                    stride, \
                    reinterpret_cast<cuComplex*>(C.data()), \
                    LDC); \
    Kokkos::Profiling::popRegion(); \
  } \
};

KOKKOSDNN_DCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSDNN_DCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSDNN_DCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSDNN_DCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSDNN_SCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSDNN_SCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSDNN_SCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSDNN_SCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSDNN_ZCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSDNN_ZCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSDNN_ZCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSDNN_ZCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSDNN_CCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSDNN_CCONV2D_CUBLAS( Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace, false)
KOKKOSDNN_CCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true)
KOKKOSDNN_CCONV2D_CUBLAS( Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

}
}
#endif // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#endif
