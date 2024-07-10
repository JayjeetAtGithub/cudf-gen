/*
 * Copyright (c) 2021-2024, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudf/aggregation.hpp>
#include <cudf/groupby.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/io/csv.hpp>
#include <cudf/table/table.hpp>
#include <cudf/column/column.hpp>
#include <cudf/strings/combine.hpp>
#include <cudf/column/column_factories.hpp>
#include <cudf/strings/convert/convert_integers.hpp>

#include <cudf/scalar/scalar.hpp>
#include <cudf/filling.hpp>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/generate.h>
#include <thrust/random.h>

#include <rmm/exec_policy.hpp>
#include <thrust/execution_policy.h>
#include <rmm/cuda_device.hpp>
#include <rmm/mr/device/cuda_memory_resource.hpp>
#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda/functional>
#include <thrust/copy.h>
#include <thrust/distance.h>
#include <thrust/equal.h>
#include <thrust/execution_policy.h>
#include <thrust/generate.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/remove.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>

#include <thrust/sequence.h>
#include <thrust/transform.h>

#include "utils.hpp"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <cudf/concatenate.hpp>



struct string_generator {
  char* chars;
  thrust::minstd_rand engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;
  string_generator(char* c, thrust::minstd_rand& engine)
    : chars(c), engine(engine), char_dist(32, 137)
  // ~90% ASCII, ~10% UTF-8.
  // ~80% not-space, ~20% space.
  // range 32-127 is ASCII; 127-136 will be multi-byte UTF-8
  {
  }
  __device__ void operator()(thrust::tuple<cudf::size_type, cudf::size_type> str_begin_end)
  {
    auto begin = thrust::get<0>(str_begin_end);
    auto end   = thrust::get<1>(str_begin_end);
    engine.discard(begin);
    for (auto i = begin; i < end; ++i) {
      auto ch = char_dist(engine);
      if (i == end - 1 && ch >= '\x7F') ch = ' ';  // last element ASCII only.
      if (ch >= '\x7F')                            // x7F is at the top edge of ASCII
        chars[i++] = '\xC4';                       // these characters are assigned two bytes
      chars[i] = static_cast<char>(ch + (ch >= '\x7F'));
    }
  }
};

template<typename T>
struct generate_random_value
{
    T lower;
    T upper;

    __host__ __device__
    generate_random_value(T lower, T upper) : lower(lower), upper(upper) {}

    __host__ __device__
        float operator()(const unsigned int idx) const
        {
            if (cudf::is_numeric<T>()) {
              thrust::default_random_engine rng;
              thrust::uniform_int_distribution<int> dist(lower, upper);
              rng.discard(idx);
              return dist(rng);
            } else {
              thrust::default_random_engine rng;
              thrust::uniform_real_distribution<float> dist(lower, upper);
              rng.discard(idx);
              return dist(rng);
            }
        }
};

template<typename T>
std::unique_ptr<cudf::column> gen_rand_col(T lower, T upper, cudf::size_type count) {
  cudf::data_type type;
  if (cudf::is_numeric<T>()) {
    type = cudf::data_type{cudf::type_id::INT32};
  } else {
    type = cudf::data_type{cudf::type_id::FLOAT64};
  }
  auto col = cudf::make_numeric_column(type, count, cudf::mask_state::UNALLOCATED, cudf::get_default_stream());
  thrust::transform(
    rmm::exec_policy(cudf::get_default_stream()),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(count),
    col->mutable_view().begin<cudf::size_type>(),
    generate_random_value<T>(lower, upper)
  );
  return col;
}


std::unique_ptr<cudf::table> generate_supplier(int32_t scale_factor) {
  cudf::size_type num_rows = 10000 * scale_factor;

  // Generate the `s_suppkey` column
  auto init_value = cudf::numeric_scalar<int32_t>(0);
  auto step_value = cudf::numeric_scalar<int32_t>(1);
  auto s_suppkey = cudf::sequence(num_rows, init_value, step_value);

  // Generate the `s_name` column
  auto indices = rmm::device_uvector<cudf::string_view>(num_rows, cudf::get_default_stream());
  auto empty_str_col = cudf::make_strings_column(indices, cudf::string_view(nullptr, 0), cudf::get_default_stream());
  auto supplier_scalar = cudf::string_scalar("Supplier#");
  auto supplier_repeat = cudf::fill(empty_str_col->view(), 0, num_rows, supplier_scalar);
  auto s_suppkey_int = cudf::strings::from_integers(s_suppkey->view());
  auto tbl_view = cudf::table_view({supplier_repeat->view(), s_suppkey_int->view()});
  auto s_name = cudf::strings::concatenate(tbl_view);

  // Generate the `s_nationkey` column
  auto s_nationkey = gen_rand_col<int>(0, 24, num_rows);

  // Generate the `s_acctbal` column
  auto s_acctbal = gen_rand_col<float>(-999.99, 9999.99, num_rows);

  // Create the `supplier` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(s_suppkey));
  columns.push_back(std::move(s_name));
  columns.push_back(std::move(s_nationkey));
  columns.push_back(std::move(s_acctbal));
  
  return std::make_unique<cudf::table>(std::move(columns));
}

int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource(&mr);

  int32_t scale_factor = 1;
  auto supplier = generate_supplier(scale_factor);

  write_parquet(
    supplier->view(), "supplier.parquet", {"s_suppkey", "s_name", "s_nationkey", "s_acctbal"});

  return 0;
}
