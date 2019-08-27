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
                             [&] (const int& j,mag_type& diff_ij) {
        
        diff_ij += APT::abs(C(i, j) - C2(i, j));

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
    typedef Kokkos::TeamPolicy<execution_space>     team_policy;
    typedef typename team_policy::member_type       member_type;

    int M = (H - R) / stride + 1;
    int N = (W - S) / stride + 1;

    ViewTypeA A("A", H, W);
    ViewTypeF F("F", R, S);
    ViewTypeC C ("C", M, N);

    ViewTypeC C_sol("C_sol", M, N); 

    uint64_t seed = Kokkos::Impl::clock_tic();
    Kokkos::Random_XorShift64_Pool<execution_space> rand_pool(seed);
   
    Kokkos::parallel_for("KokkosDNN::Test::Filling_A",
                         team_policy(H, Kokkos::AUTO(), 16),
      KOKKOS_LAMBDA (const member_type &team_member) {

        int i = team_member.league_rank();

        Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member, W),
          [=] (const int &j) {

            A(i,j) = i + j;
      });
    });

    Kokkos::parallel_for("KokkosDNN::Test::Filling_F",
                         team_policy(R, Kokkos::AUTO(), 16),
      KOKKOS_LAMBDA (const member_type &team_member) {

        int i = team_member.league_rank();

        Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member, S),
          [=] (const int &j) {

            F(i,j) = 1;
      });
    });

    Kokkos::fill_random(C, rand_pool, ScalarC(10));

    Kokkos::parallel_for("KokkosDNN::Test::Filling_Csol",
                         team_policy(M, Kokkos::AUTO(), 16),
      KOKKOS_LAMBDA (const member_type &team_member) {

        int i = team_member.league_rank();

        Kokkos::parallel_for (Kokkos::TeamThreadRange(team_member, N),
          [=] (const int &j) {

            C_sol(i,j) = R * S * (i + (R / 2) + j + (S / 2));
      });
    });


    Kokkos::fence();

    // OUTPUT
/*
    // A
    std::cout << "\nImage A: \n";
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
/*
    // C_sol
    std::cout << "\nSolution C_sol: \n";
    for (int i = 0; i < M; ++i) {
      std::cout << "\nRow: " << i << " ";
      
      for (int j = 0; j < N; ++j) {
        std::cout << " " << C_sol(i, j);

      }
      std::cout << std::endl;
    }
*/
    // !OUTPUT
    
    std::cout << "\nAbout to conv" << std::endl;

    struct timeval begin, end;

    gettimeofday(&begin, NULL);

    KokkosDNN::conv2d(A, F, stride, C);

    Kokkos::fence();

    gettimeofday(&end, NULL);

    double inner_t = 1.0 * (end.tv_sec - begin.tv_sec) +
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nInner Time: " << inner_t << std::endl;

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


    if (M != 0 && N != 0 && H != 0 && W != 0) {
       
      double diff_C_average = diff_C / (M * N);
      
      std::cout << "\n(M,N): " << M << " " << N << " DIFF_C_AVERAGE: " << diff_C_average 
        << " diff_C: " << diff_C << std::endl;


      EXPECT_TRUE(diff_C_average < 1);
      EXPECT_TRUE(diff_C_average < .0001);
    }
  }
}

template<class ScalarA, class ScalarF, class ScalarC, class Device>
int test_conv2d(int stride) {

    struct timeval begin, end;

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutLeft, Device> view_type_a_ll;
  typedef Kokkos::View<ScalarF**, Kokkos::LayoutLeft, Device> view_type_f_ll;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutLeft, Device> view_type_c_ll;
  
  // Teams: N, Image: H x W, Filter: R x S, Stride) {
/*
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


//---------------------------------------------------------------------------//
 
    std::cout << "\nLayout Left" << std::endl;
     
    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                           view_type_c_ll, Device>(1024, 1024, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t1024_ll = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (1024, 1024), F = (5,5), stride = " 
        << stride << ": " << t1024_ll << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                           view_type_c_ll, Device>(4096, 4096, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t4096_ll = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (4096, 4096), F = (5,5), stride = " 
        << stride << ": " << t4096_ll << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
                           view_type_c_ll, Device>(16384, 16384, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t16384_ll = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (16384, 16384), F = (5,5), stride = " 
        << stride << ": " << t16384_ll << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

//    Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
//                           view_type_c_ll, Device>(20000, 20000, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t20000_ll = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (20000, 20000), F = (5,5), stride = " 
        << stride << ": " << t20000_ll << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

//    Test::impl_test_conv2d<view_type_a_ll, view_type_f_ll, 
//                           view_type_c_ll, Device>(65536, 65536, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t65536_ll = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (65536, 65536), F = (5,5), stride = " 
        << stride << ": " << t65536_ll << std::endl;

//---------------------------------------------------------------------------//

#endif

  
  
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || (!defined(KOKKOSKERNELS_ETI_ONLY) && !defined(KOKKOSKERNELS_IMPL_CHECK_ETI_CALLS))
  typedef Kokkos::View<ScalarA**, Kokkos::LayoutRight, Device> view_type_a_lr;
  typedef Kokkos::View<ScalarF**, Kokkos::LayoutRight, Device> view_type_f_lr;
  typedef Kokkos::View<ScalarC**, Kokkos::LayoutRight, Device> view_type_c_lr;
  
  // Teams: N, Image: H x W, Filter: R x S, Stride) {
/*
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(0,       0, 0, 0, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(16,     16, 3, 3, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(179,    15, 5, 3, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(12,   1071, 3, 5, stride);
  Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                         view_type_c_lr, Device>(1024, 1024, 5, 5, stride);
*/

//---------------------------------------------------------------------------//

    std::cout << "\nLayout Right" << std::endl;

    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                           view_type_c_lr, Device>(1024, 1024, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t1024_lr = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (1024, 1024), F = (5,5), stride = " 
        << stride << ": " << t1024_lr << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                           view_type_c_lr, Device>(4096, 4096, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t4096_lr = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (4096, 4096), F = (5,5), stride = " 
        << stride << ": " << t4096_lr << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                           view_type_c_lr, Device>(16384, 16384, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t16384_lr = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (16384, 16384), F = (5,5), stride = " 
        << stride << ": " << t16384_lr << std::endl;

//---------------------------------------------------------------------------//
  
    gettimeofday(&begin, NULL);

    Test::impl_test_conv2d<view_type_a_lr, view_type_f_lr, 
                           view_type_c_lr, Device>(65536, 65536, 5, 5, stride);

    gettimeofday(&end, NULL);

    double t65536_lr = 1.0 * (end.tv_sec - begin.tv_sec) + 
        1.0e-6 * (end.tv_usec - begin.tv_usec);

    std::cout << "\nTime A = (65536, 65536), F = (5,5), stride = " 
        << stride << ": " << t65536_lr << std::endl;

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
    test_conv2d<float, float, float, TestExecSpace> (1);
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
//    test_conv2d<double, double, double, TestExecSpace> (1);
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
