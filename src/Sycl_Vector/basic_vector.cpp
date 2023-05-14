#ifndef BASIC_VECTOR_CPP
#define BASIC_VECTOR_CPP
////////////////////////////////////////////////////////////////////////////
/// \file
/// \brief Header containing a vector class with various mathematical
//         operations that are executed in parallel
///////////////////////////////////////////////////////////////////////////

#include <vector>
#include <chrono>
#include <thread>

// Pybind11
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

// Sycl
#include <CL/sycl.hpp>

namespace py = pybind11;

///////////////////////////////////////////////////////////////////////////
/// \defgroup Sycl Vector
/// \brief    Creates a sycl based vector class
/// @{
///////////////////////////////////////////////////////////////////////////
/// \brief Creates a vector with operations for sycl based computations

class Basic_Sycl_Vector{
  ////////////////////////////////////////////////////////////////////////
  /// \brief Selected device
  sycl::queue Q;

  ////////////////////////////////////////////////////////////////////////
  /// \brief Vector size
  size_t SIZE;

  ////////////////////////////////////////////////////////////////////////
  /// \brief Vector
  std::vector<double> A;

  public:
    ////////////////////////////////////////////////////////////////////////
    /// \brief Prints the selected device
    void print_device();

    ////////////////////////////////////////////////////////////////////////
    /// \brief Selects the gpu for the device
    void select_gpu_device();

    ////////////////////////////////////////////////////////////////////////
    /// \brief Sets all vector elements to zero
    void reset();

    ////////////////////////////////////////////////////////////////////////
    /// \brief Adds some value x to each element
    template<typename Scalar_type>
    void add_each_element(Scalar_type x);

    ////////////////////////////////////////////////////////////////////////
    /// \brief Subtracts some value x from each element
    template<typename Scalar_type>
    void subtract_each_element(Scalar_type x);

    ////////////////////////////////////////////////////////////////////////
    /// \brief Multiplies each element by some value x
    template<typename Scalar_type>
    void multiply_each_element(Scalar_type x);

      ////////////////////////////////////////////////////////////////////////
    /// \brief Multiplies each element by some value x
    template<typename Scalar_type>
    void divide_each_element(Scalar_type x);

    ////////////////////////////////////////////////////////////////////////
    /// \brief Returns the vector
    std::vector<double> get_vector();

    ////////////////////////////////////////////////////////////////////////
    /// \brief Constructor that initializes the vector
    Basic_Sycl_Vector(int SIZE_in): SIZE(SIZE_in){
      for(int i = 0; i < SIZE; ++i){
        A.push_back(0.0);
      }
    }
};

/// @}
// end "Basic Sycl Vector" doxygen group

////////////////////////////////////////////////////////////////////////
void Basic_Sycl_Vector::print_device(){
  std::cout << "DEVICE: "
            << Q.get_device().template get_info<sycl::info::device::name>()
            << "\nVENDOR: "
            << Q.get_device().template get_info<sycl::info::device::vendor>()
            << "\n" << std::endl;
}

////////////////////////////////////////////////////////////////////////
void Basic_Sycl_Vector::select_gpu_device(){
  sycl::queue Q{sycl::gpu_selector_v};
}

////////////////////////////////////////////////////////////////////////
void Basic_Sycl_Vector::reset(){
  // creating a sycl scope
  {
    // creating a buffer for the vector
    sycl::buffer<double> A_buffer{A};

    // executing a sycl kernel
    Q.submit([&](sycl::handler &h){
      // creating a device accessor
      sycl::accessor A_access{A_buffer, h};
      h.parallel_for<Basic_Sycl_Vector>(SIZE, [=](sycl::id<1> idx){
        const int i = 0;
        A[i] = 0.0;
      });
    });
  }
}

////////////////////////////////////////////////////////////////////////
template<typename Scalar_type>
void Basic_Sycl_Vector::add_each_element(Scalar_type x){
  // creating a sycl scope
  {
    // creating a buffer for the vector
    sycl::buffer<double> A_buffer{A};

    // executing a sycl kernel
    Q.submit([&](sycl::handler &h){
      // creating a device accessor
      sycl::accessor A_access{A_buffer, h};
      h.parallel_for<Basic_Sycl_Vector>(SIZE, [=](sycl::id<1> idx){
        const int i = 0;
        A_access[i] += x;
      });
    });
  }
}

////////////////////////////////////////////////////////////////////////
template<typename Scalar_type>
void Basic_Sycl_Vector::subtract_each_element(Scalar_type x){
  // creating a sycl scope
  {
    // creating a buffer for the vector
    sycl::buffer<double> A_buffer{A};

    // executing a sycl kernel
    Q.submit([&](sycl::handler &h){
      // creating a device accessor
      sycl::accessor A_access{A_buffer, h};
      h.parallel_for<Basic_Sycl_Vector>(SIZE, [=](sycl::id<1> idx){
        const int i = 0;
        A_access[i] -= x;
      });
    });
  }
}

////////////////////////////////////////////////////////////////////////
template<typename Scalar_type>
void Basic_Sycl_Vector::multiply_each_element(Scalar_type x){
  // creating a sycl scope
  {
    // creating a buffer for the vector
    sycl::buffer<double> A_buffer{A};

    // executing a sycl kernel
    Q.submit([&](sycl::handler &h){
      // creating a device accessor
      sycl::accessor A_access{A_buffer, h};
      h.parallel_for<Basic_Sycl_Vector>(SIZE, [=](sycl::id<1> idx){
        const int i = 0;
        A_access[i] *= x;
      });
    });
  }
}

////////////////////////////////////////////////////////////////////////
template<typename Scalar_type>
void Basic_Sycl_Vector::divide_each_element(Scalar_type x){
  // creating a sycl scope
  {
    // creating a buffer for the vector
    sycl::buffer<double> A_buffer{A};

    // executing a sycl kernel
    Q.submit([&](sycl::handler &h){
      // creating a device accessor
      sycl::accessor A_access{A_buffer, h};
      h.parallel_for<Basic_Sycl_Vector>(SIZE, [=](sycl::id<1> idx){
        const int i = 0;
        A_access[i] /= x;
      });
    });
  }
}

////////////////////////////////////////////////////////////////////////
std::vector<double> Basic_Sycl_Vector::get_vector(){
  return A;
}

#endif //#ifndef BASIC_VECTOR_CPP


PYBIND11_MODULE(sycl_vector, m){
  m.doc() = R"myDelim(
    User guide documentation for the 'Basic Sycl Vector' module.
    ------------------------------------------------------------

    .. currentmodule:: basic sycl vector

    .. autosummary::
      :toctree: _generate

      print_device
      select_gpu_device
      reset
      add_each_element
      subtract_each_element
      multiply_each_element
      divide_each_element

  )myDelim";

  py::class_<Basic_Sycl_Vector>(m, "basic_sycl_vector").def(py::init<int>(), R"myDelim(
    Initialize a basic sycl vector with some input size 'SIZE'

    Parameters
    ----------
    SIZE
  )myDelim").def("print_device", &Basic_Sycl_Vector::print_device, R"myDelim(
    Prints the selected device for SYCL queue
  )myDelim").def("select_gpu_device", &Basic_Sycl_Vector::select_gpu_device, R"myDelim(
    Selects GPU for SYCL queue
  )myDelim").def("reset", &Basic_Sycl_Vector::reset, R"myDelim(
    Resets every vector input to be zero
  )myDelim").def("add_each_element", &Basic_Sycl_Vector::add_each_element<double>, R"myDelim(
    Adds a specific value x to each vector element
  )myDelim").def("subtract_each_element", &Basic_Sycl_Vector::subtract_each_element<double>, R"myDelim(
    Subtracts a specific value x to each vector element
  )myDelim").def("multiply_each_element", &Basic_Sycl_Vector::multiply_each_element<double>, R"myDelim(
    Multiplies a specific value x to each vector element
  )myDelim").def("divide_each_element", &Basic_Sycl_Vector::divide_each_element<double>, R"myDelim(
    Divides a specific value x to each vector element
  )myDelim");
}