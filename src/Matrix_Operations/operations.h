#ifndef OPERATIONS_H
#define OPERATIONS_H

////////////////////////////////////////////////////////////////////////////
// This file is part of a simple test project titled Doc Test Projects
////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Header containing the definition of matrix operations
///////////////////////////////////////////////////////////////////////////

#include <CL/sycl.hpp>
#include <vector>

/////////////////////////////////////////////////////////////////////////
/// \brief Executes a matrix addition.
/// \return Returns the sum of the two matrices.
template<typename Queue_type>
void print_device(Queue_type& Q){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

/////////////////////////////////////////////////////////////////////////
/// \brief Executes a matrix addition.
/// \return Returns the sum of the two matrices.
template<typename Queue_type, typename Scalar_type>
auto matrix_add(Queue_type Q,
                std::vector<Scalar_type> A,
                std::vector<Scalar_type> B,
                size_t M,
                size_t N){

  std::vector<Scalar_type> C(N);

  // sycl scope
  {
    // memory buffers
    sycl::buffer<Scalar_type, 2> A_buf(A.data(), sycl::range<2>(M, N));
    sycl::buffer<Scalar_type, 2> B_buf(B.data(), sycl::range<2>(M, N));
    sycl::buffer<Scalar_type, 2> C_buf(C.data(), sycl::range<2>(M, N));

    Q.submit([&](sycl::handler &h){
      // memory accessors
      sycl::accessor<Scalar_type> A_acc{A_buf, h};
      sycl::accessor<Scalar_type> B_acc{B_buf, h};
      sycl::accessor<Scalar_type> C_acc{C_buf, h};
      h.parallel_for(sycl::range{M, N}, [=](sycl::id<2> idx){
        C_acc[idx] = A_acc[idx] + B_acc[idx];
      });
    });
  }

  return C;
}

/////////////////////////////////////////////////////////////////////////
/// \brief Executes a matrix multiplication.
/// \return Returns the product of the two matrices.
template<typename Queue_type, typename Scalar_type>
auto matrix_multiplication(Queue_type Q,
                           std::vector<Scalar_type> A,
                           std::vector<Scalar_type> B,
                           size_t M,
                           size_t N,
                           size_t K){

  std::vector<Scalar_type> C(N);

  // sycl scope
  {
    // memory buffers
    sycl::buffer<Scalar_type, 2> A_buf(A.data(), sycl::range<2>(M, N));
    sycl::buffer<Scalar_type, 2> B_buf(B.data(), sycl::range<2>(N, K));
    sycl::buffer<Scalar_type, 2> C_buf(C.data(), sycl::range<2>(M, K));

    Q.submit([&](sycl::handler &h){
      // memory accessors
      sycl::accessor<Scalar_type> A_acc{A_buf, h};
      sycl::accessor<Scalar_type> B_acc{B_buf, h};
      sycl::accessor<Scalar_type> C_acc{C_buf, h};
      h.parallel_for(sycl::range{M, K}, [=](sycl::id<2> idx){
        const int i = idx[0];
        const int j = idx[1];
        for(int k = 0; k < N; ++k){
          C_acc[j][i] += A_acc[j][k] + B_acc[k][i];
        }
      });
    });
  }

  return C;
}

#endif //#ifndef OPERATIONS_H