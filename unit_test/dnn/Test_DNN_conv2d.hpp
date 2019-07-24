#include<gtest/gtest.h>
#include<Kokkos_Core.hpp>

#include<iostream>
#include<sstream>
#include<sys/time.h>
#include "KokkosKernels_helpers.hpp"

#include "Kokkos_Random.hpp"
#include "KokkosDNN_conv2d.hpp"
#include "KokkosKernels_TestUtils.hpp"

namespace Test {

/*
  template<class ViewTypeA, class ViewTypeF, 
           class ViewTypeC, class ExecutionSpace>
  struct VanillaCONV2D {

    int H, W, R, S, M, N;
    ViewTypeA A;
    ViewTypeF F;
    ViewTypeC C;

    typedef typename ViewTypeA::value_type ScalarA;
    typedef typename ViewTypeF::value_type ScalarF;
    typedef typename ViewTypeC::value_type ScalarC;
//    typedef typename APT::mag_type mag_type;
    int stride;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::
                     member_type& team) const {

// GNU COMPILER BUG WORKAROUND
#if defined(KOKKOS_COMPILER_GNU) && !defined(__CUDA_ARCH__)
      int i = team.league_rank();
#else
      const int i = team.league_rank();
#endif
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), 
                          [&] (const int& j) {
        ScalarC C_ij = 0.0;

        // GNU 5.3, 5.4 and 6.1 (and maybe more) crash with another nested 
        // lambda here

        // Perform convolution for element C_ij
#if defined(KOKKOS_COMPILER_GNU) && !defined(KOKKOS_COMPILER_NVCC)
        for (int r = 0; r < R; r++) {
          for (int s = 0; s < S; s++) {
            ScalarA A_ij_rs = A(i + r, j + s);
            ScalarF F_rs    = F(r, s);

            C_ij += A_ij_rs * F_rs;
          }
        }
#else
        Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team, R * S), 
                               [&] (const int& rs, ScalarC& lsum) {

          ScalarA A_ij_rs = A(i + rs % R, j + rs / R);
          ScalarF F_rs    = F(rs % R, rs / R);

          lsum += A_ij_rs * F_rs;
        }, C_ij);
#endif

        C(i,j) = C(i,j) + C_ij;
      });
    }
  };
*/

  template<class ViewTypeC, class ExecutionSpace>
  struct DiffConv2d {
    int M, N;
    ViewTypeC C, C2;

    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;

    KOKKOS_INLINE_FUNCTION
    void operator() (const typename Kokkos::TeamPolicy<ExecutionSpace>::
                     member_type& team, mag_type& diff) const {
      const int i = team.league_rank();
      mag_type diff_row = 0;

      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, N), 
            KOKKOS_LAMBDA (const int& j, mag_type& diff_ij) {
//        printf("A (%l %l) (%l %l) (%i %i)\n", 
//               C.extent(0), C.extent(1), 
//               C2.extent(0), C2.extent(1), i, j);
        
        diff_ij += APT::abs(C(i, j) - C2(i, j));

//        printf("F (%l %l) (%l %l) (%i %i)\n", 
//               C.extent(0), C.extent(1), 
//               C2.extent(0), C2.extent(1), i, j);
      }, diff_row);

      Kokkos::single(Kokkos::PerTeam(team), [&] () {
        diff += diff_row;
      });
    }
  };

  template<class ViewTypeA, class ViewTypeF, class ViewTypeC, class Device>
  void impl_test_conv2d(int H, int W, int R, int S, int stride) {

    typedef typename ViewTypeA::device_type::execution_space execution_space;
//    typedef typename ViewTypeA::value_type ScalarA;
//    typedef typename ViewTypeF::value_type ScalarF;
    typedef typename ViewTypeC::value_type ScalarC;
    typedef Kokkos::Details::ArithTraits<ScalarC> APT;
    typedef typename APT::mag_type mag_type;

//    double machine_eps = APT::epsilon();

    int M = (H - R) / stride + 1;
    int N = (W - S) / stride + 1;

    ViewTypeA A("A", H, W);
    ViewTypeF F("F", R, S);
    ViewTypeC C ("C", M, N);
 //   ViewTypeC C2("C2", M, N);
    ViewTypeC C_sol("C_sol", M, N); 

    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);

//    Kokkos::TeamPolicy<execution_space> team(M, Kokkos::AUTO, 16);

//    Kokkos::fill_random(A, rand_pool, ScalarA(10));
//    Kokkos::fill_random(F, rand_pool, ScalarF(1));
   
//    for (int i = 0; i < H; ++i) {
    Kokkos::parallel_for("Filling A", H,  
//                         Kokkos::TeamPolicy<execution_space>(H, 
//                            Kokkos::AUTO, 16), 
        KOKKOS_LAMBDA (const int& i) {
            for (int j = 0; j < W; ++j) {                     
              A(i,j) = i + j;
            }
        });

    std::cout << "\nSet the A" << std::endl;

//    });
//    team.team_barrier();


//    for (int i = 0; i < R; ++i) {
    Kokkos::parallel_for("Filling F", R,
//                         Kokkos::TeamPolicy<execution_space>(R, 
//                            Kokkos::AUTO, 16), 
        KOKKOS_LAMBDA (const int& i) {
            for (int j = 0; j < S; ++j) {
              F(i,j) = 1;
            }
        });

//    });
//    team.team_barrier();

    // ??? Junk
    Kokkos::fill_random(C, rand_pool, ScalarC(10));
//    Kokkos::deep_copy(C2, C);

    // C3(i,j) = #elements in F, if all elements of filter and image are 1
//    for (int i = 0; i < M; ++i) {
    Kokkos::parallel_for("Filling C_Sol", M,
//                         Kokkos::TeamPolicy<execution_space>(M, 
//                            Kokkos::AUTO, 16), 
        KOKKOS_LAMBDA (const int& i) { 
            for (int j = 0; j < N; ++j) { 
              C_sol(i,j) = R * S * (i + (R / 2) + j + (S / 2));
            }
        });
//    })
//    team.team_barrier();

    Kokkos::fence();

    // OUTPUT

    // A
/*    std::cout << "\nImage A: \n";
    for (int i = 0; i < H; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < W; ++j) {
        std::cout << " " << A(i, j);

      }
      std::cout << std::endl;
    }

    // F
    std::cout << "\nFilter F: \n";
    for (int i = 0; i < R; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < S; ++j) {
        std::cout << " " << F(i, j);

      }
      std::cout << std::endl;
    }


    // C
    std::cout << "\nRandom C: \n";
    for (int i = 0; i < M; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < N; ++j) {
        std::cout << " " << C(i, j);

      }
      std::cout << std::endl;
    }
*/

/*    // C2
    std::cout << "\nRandom C2: \n";
    for (int i = 0; i < M; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < N; ++j) {
        std::cout << " " << C2(i, j);

      }
      std::cout << std::endl;
    }
*/

    // C_sol
/*    std::cout << "\nSolution C_sol: \n";
    for (int i = 0; i < M; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < N; ++j) {
        std::cout << " " << C_sol(i, j);

      }
      std::cout << std::endl;
    }
*/

    // !OUTPUT
/*
    struct VanillaCONV2D<ViewTypeA, ViewTypeF, 
                         ViewTypeC, execution_space> vconv2d;

    vconv2d.H = H;    vconv2d.W = W;     
    vconv2d.R = R;    vconv2d.S = S;

    vconv2d.A = A;     vconv2d.F = F;
    vconv2d.C = C2;
    vconv2d.stride = stride;

    Kokkos::parallel_for("KokkosDNN::Test::VanillaCONV2D", 
                         Kokkos::TeamPolicy<execution_space>(M, 
                            Kokkos::AUTO, 16), 
                         vconv2d);
*/

    KokkosDNN::conv2d(A, F, stride, C);

    Kokkos::fence();


/*
    std::cout << "\nSolution C: \n";
    for (int i = 0; i < M; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < N; ++j) {
        std::cout << " " << C(i, j);

      }
      std::cout << std::endl;
    }
*/
/*
    std::cout << "\nSolution C2: \n";
    for (int i = 0; i < M; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < N; ++j) {
        std::cout << " " << C2(i, j);

      }
      std::cout << std::endl;
    }
*/





    Kokkos::fence();

    // Difference between C (kokkos-kernels) and C_sol (true)
    mag_type diff_C = 0;
    struct DiffConv2d<ViewTypeC, execution_space> diffconv2d_C;
    diffconv2d_C.M = M;
    diffconv2d_C.N = N;
    diffconv2d_C.C = C;
    diffconv2d_C.C2 = C_sol;


    Kokkos::parallel_reduce("KokkosDNN::Test::DiffConv2d_C", 
                            Kokkos::TeamPolicy<execution_space>(M, 
                              Kokkos::AUTO, 16), 
                            diffconv2d_C, diff_C);

/*
    // Difference between C2 (vanilla) and C3 (true)
    mag_type diff_C2 = 0;
    struct DiffConv2d<ViewTypeC, execution_space> diffconv2d_C2;
    diffconv2d_C2.M = M;
    diffconv2d_C2.N = N;
    diffconv2d_C2.C = C2;
    diffconv2d_C2.C2 = C3;



    Kokkos::parallel_reduce("KokkosDNN::Test::DiffConv2d_C2", 
                            Kokkos::TeamPolicy<execution_space>(M, 
                              Kokkos::AUTO, 16), 
                            diffconv2d_C2, diff_C2);
*/

    if (M != 0 && N != 0 && H != 0 && W != 0) {
       
      double diff_C_average = diff_C / (M * N);
//      double diff_C2_average = diff_C2 / (M * N);
/*
      // Expected Result: Random Walk in the least significant bit (i.e. 
      // ~ sqrt(W)*eps 
      // eps scales with the total sum and has a factor in it for the 
      // accuracy of the operations ->
      // eps = K * 75 * machine_eps * 7
      double diff_C_expected = 1.0 * sqrt(W) * W * 75 * machine_eps * 7;

      printf("Result: %e %e\n", diff_C_average, diff_C_expected);
      EXPECT_TRUE(diff_C_average < 1.05 * diff_C_expected);
*/

//      std::ostringstream os;

      std::cout << "\n(M,N): " << M << " " << N << " DIFF_C_AVERAGE: " << diff_C_average 
        << " diff_C: " << diff_C << std::endl;

//      std::cout << "\n(M,N): " << M << " " << N << " DIFF_C2_AVERAGE: " << diff_C2_average 
//        << " diff_C2: " << diff_C2 << std::endl;


      EXPECT_TRUE(diff_C_average < 1);
    }
  }
}

template<class ScalarA, class ScalarF, class ScalarC, class Device>
int test_conv2d(int stride) {

  struct timeval begin, end;


#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
/*
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarF**, Kokkos::LayoutLeft, Device> view_type_f_ll;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutLeft, Device> view_type_c_ll;
  
  // Teams: N, Image: H x W, Filter: R x S, Stride) {
  Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                         view_type_c_ll, Device>(0,       0, 0, 0, stride);
  Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                         view_type_c_ll, Device>(16,     16, 3, 3, stride);
  Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                         view_type_c_ll, Device>(179,    15, 5, 3, stride);
  Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                         view_type_c_ll, Device>(12,   3071, 3, 5, stride);
  Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                         view_type_c_ll, Device>(1024, 1024, 5, 5, stride);
*/
#endif

  
  
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))

  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarF**, Kokkos::LayoutRight, Device> view_type_f_lr;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutRight, Device> view_type_c_lr;
/*  
  // Teams: N, Image: H x W, Filter: R x S, Stride) {
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(0,       0, 0, 0, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(16,     16, 3, 3, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(179,    15, 5, 3, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(12,   1071, 3, 5, stride);
*/
  
//---------------------------------------------------------------------------//
  
  gettimeofday(&begin, NULL);

  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(1024, 1024, 5, 5, stride);

  gettimeofday(&end, NULL);

  double t1024 = 1.0 * (end.tv_sec - begin.tv_sec) + 
      1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::cout << "\nTime A = (1024, 1024), F = (5,5), stride = " << stride << ": " << t1024 << std::endl;

//---------------------------------------------------------------------------//

  gettimeofday(&begin, NULL);
  
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(4096, 4096, 5, 5, stride);

  gettimeofday(&end, NULL);

  double t4096 = 1.0 * (end.tv_sec - begin.tv_sec) + 
      1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::cout << "\nTime A = (4096, 4096), F = (5,5), stride = " << stride << ": " << t4096 << std::endl;

//---------------------------------------------------------------------------//

  gettimeofday(&begin, NULL);

  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(16384, 16384, 5, 5, stride);

  gettimeofday(&end, NULL);

  double t16384 = 1.0 * (end.tv_sec - begin.tv_sec) + 
      1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::cout << "\nTime A = (16384, 16384), F = (5,5), stride = " << stride << ": " << t16384 << std::endl;

//---------------------------------------------------------------------------//

  gettimeofday(&begin, NULL);

  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(65536, 65536, 5, 5, stride);


  gettimeofday(&end, NULL);

  double t65536 = 1.0 * (end.tv_sec - begin.tv_sec) + 
      1.0e-6 * (end.tv_usec - begin.tv_usec);

  std::cout << "\nTime A = (65536, 65536), F = (5,5), stride = " << stride << ": " << t65536 << std::endl;

//---------------------------------------------------------------------------//




#endif

/*
#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutStride, 
                       Device> view_type_a_ls;
  typedef Kokkos::View<ScalarX*, Kokkos::LayoutStride, Device> view_type_b_ls;
  typedef Kokkos::View<ScalarY*, Kokkos::LayoutStride, Device> view_type_c_ls;
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, 
                       view_type_c_ls, Device>(mode,0,1024);
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, 
                       view_type_c_ls, Device>(mode,13,1024);
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, 
                       view_type_c_ls, Device>(mode,1024,1024);
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ls, 
                       view_type_c_ls, Device>(mode,132231,1024);
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
  Test::impl_test_gemv<view_type_a_ls, view_type_b_ll, 
                       view_type_c_lr, Device>(mode,1024,1024);
  Test::impl_test_gemv<view_type_a_ll, view_type_b_ls, 
                       view_type_c_lr, Device>(mode,1024,1024);
#endif
*/

  return 1;
}

#if defined(KOKKOSKERNELS_INST_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, conv2d_float ) {
  Kokkos::Profiling::pushRegion("KokkosDNN::Test::conv2d_float");

    // Vary convolution stride
//    test_conv2d<float, float, float, TestExecSpace> (1);
//    test_conv2d<float, float, float, TestExecSpace> (2);
//    test_conv2d<float, float, float, TestExecSpace> (3);
//    test_conv2d<float, float, float, TestExecSpace> (4);

  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, conv2d_double ) {
  Kokkos::Profiling::pushRegion("KokkosDNN::Test::conv2d_double");
 
    // Vary convolution stride
    test_conv2d<double, double, double, TestExecSpace> (1);
//    test_conv2d<double, double, double, TestExecSpace> (2);
//    test_conv2d<double, double, double, TestExecSpace> (3);
//    test_conv2d<double, double, double, TestExecSpace> (4);
  
  Kokkos::Profiling::popRegion();
}
#endif


#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, conv2d_complex_double ) {
  Kokkos::Profiling::pushRegion("KokkosDNN::Test::conv2d_complex_double");

    // Vary convolution stride
//    test_conv2d<Kokkos::complex<double>, Kokkos::complex<double>, 
//                Kokkos::complex<double>, TestExecSpace> (1);
//    test_conv2d<Kokkos::complex<double>, Kokkos::complex<double>, 
//                Kokkos::complex<double>, TestExecSpace> (2);
//    test_conv2d<Kokkos::complex<double>, Kokkos::complex<double>, 
//                Kokkos::complex<double>, TestExecSpace> (3);
//    test_conv2d<Kokkos::complex<double>, Kokkos::complex<double>, 
//                Kokkos::complex<double>, TestExecSpace> (4);
 
  Kokkos::Profiling::popRegion();
}
#endif

#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, conv2d_complex_float ) {
  Kokkos::Profiling::pushRegion("KokkosDNN::Test::conv2d_complex_float");
 
    // Vary convolution stride
//    test_conv2d<Kokkos::complex<float>, Kokkos::complex<float>, 
//                Kokkos::complex<float>, TestExecSpace> (1);
//    test_conv2d<Kokkos::complex<float>, Kokkos::complex<float>, 
//                Kokkos::complex<float>, TestExecSpace> (2);
//    test_conv2d<Kokkos::complex<float>, Kokkos::complex<float>, 
//                Kokkos::complex<float>, TestExecSpace> (3);
//    test_conv2d<Kokkos::complex<float>, Kokkos::complex<float>, 
//                Kokkos::complex<float>, TestExecSpace> (4);

  Kokkos::Profiling::popRegion();
}
#endif



#if defined(KOKKOSKERNELS_INST_INT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
TEST_F( TestCategory, conv2d_int ) {
//    test_conv2d<int,int,int,TestExecSpace> (1);
}
#endif
/*
#if !defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS)
TEST_F( TestCategory, conv2d_double_int ) {
    test_conv2d<double,int,float,TestExecSpace> (1);
}
#endif
*/

/*
#define EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE) \
TEST_F( TestCategory, graph ## _ ## graph_color_d2 ## _ ## SCALAR ## _ ## ORDINAL ## _ ## OFFSET ## _ ## DEVICE ) { \
  test_coloring_d2<SCALAR,ORDINAL,OFFSET,DEVICE>(50000, 50000 * 30, 200, 10); \
  test_coloring_d2<SCALAR,ORDINAL,OFFSET,DEVICE>(50000, 50000 * 30, 100, 10); \
}

#if defined(KOKKOSKERNELS_INST_DOUBLE)
#if (defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
     && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
 EXECUTE_TEST(double, int, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
  && defined (KOKKOSKERNELS_INST_OFFSET_INT) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  EXECUTE_TEST(double, int64_t, int, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_ORDINAL_INT) \
  && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  EXECUTE_TEST(double, int, size_t, TestExecSpace)
#endif

#if (defined (KOKKOSKERNELS_INST_ORDINAL_INT64_T) \
  && defined (KOKKOSKERNELS_INST_OFFSET_SIZE_T) ) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  EXECUTE_TEST(double, int64_t, size_t, TestExecSpace)
#endif
#endif
*/
