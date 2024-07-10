# Copyright (c) 2020-2024, NVIDIA CORPORATION.

cmake_minimum_required(VERSION 3.26.4)

include(./set_cuda_architecture.cmake)

# initialize cuda architecture
rapids_cuda_init_architectures(dbgen)
rapids_cuda_set_architectures(RAPIDS)

project(
  dbgen
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

include(./fetch_dependencies.cmake)

# Configure your project here
add_executable(dbgen src/dbgen.cu)

set_property(TARGET dbgen 
             PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(dbgen PUBLIC cudf::cudf)
target_compile_features(dbgen PUBLIC cxx_std_17 cuda_std_17)
target_compile_options(dbgen PUBLIC $<$<COMPILE_LANG_AND_ID:CUDA,NVIDIA>:--expt-relaxed-constexpr>)
