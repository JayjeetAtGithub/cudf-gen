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



struct generate_random_string {
  char* chars;
  thrust::default_random_engine engine;
  thrust::uniform_int_distribution<unsigned char> char_dist;
  
  __host__ __device__
  generate_random_string(char* c) : chars(c), char_dist(32, 137) {}
  
  __host__ __device__ void operator()(thrust::tuple<cudf::size_type, cudf::size_type> str_begin_end)
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
              thrust::default_random_engine engine;
              thrust::uniform_int_distribution<int> dist(lower, upper);
              engine.discard(idx);
              return dist(engine);
            } else {
              thrust::default_random_engine engine;
              thrust::uniform_real_distribution<float> dist(lower, upper);
              engine.discard(idx);
              return dist(engine);
            }
        }
};

// lengths: [10, 20, 30, 40, 50]
// offsets: [0, 10, 30, 60, 100, 150]
void gen_rand_string_col(cudf::size_type num_rows)
{  
  rmm::device_uvector<cudf::size_type> lengths(num_rows, cudf::get_default_stream());
  rmm::device_uvector<cudf::size_type> offsets(num_rows + 1, cudf::get_default_stream());
  
  thrust::transform(
    rmm::exec_policy(cudf::get_default_stream()),
    thrust::make_counting_iterator(0),
    thrust::make_counting_iterator(num_rows),
    lengths.begin(),
    generate_random_value<cudf::size_type>(10, 40)
  );

  thrust::exclusive_scan(
    rmm::exec_policy(cudf::get_default_stream()),
    lengths.begin(),
    lengths.end(),
    offsets.begin()
  );

  cudf::size_type total_length = *thrust::device_pointer_cast(offsets.end() - 1);
  rmm::device_uvector<char> chars(total_length, cudf::get_default_stream());  
  generate_random_string(chars.data());
  std::cout << "here" << std::endl;
}

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

std::unique_ptr<cudf::table> generate_nation(int32_t scale_factor) {
  cudf::size_type num_rows = 25;

  // Generate the `n_nationkey` column
  auto init_value = cudf::numeric_scalar<int32_t>(0);
  auto step_value = cudf::numeric_scalar<int32_t>(1);
  auto n_nationkey = cudf::sequence(num_rows, init_value, step_value);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(n_nationkey));
  return std::make_unique<cudf::table>(std::move(columns));
}

std::unique_ptr<cudf::table> generate_region(int32_t scale_factor) {
  cudf::size_type num_rows = 5;

  // Generate the `r_regionkey` column
  auto init_value = cudf::numeric_scalar<int32_t>(0);
  auto step_value = cudf::numeric_scalar<int32_t>(1);
  auto r_regionkey = cudf::sequence(num_rows, init_value, step_value);

  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(r_regionkey));
  return std::make_unique<cudf::table>(std::move(columns));
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
  auto s_suppkey_str = cudf::strings::from_integers(s_suppkey->view());
  auto s_name_parts = cudf::table_view({supplier_repeat->view(), s_suppkey_str->view()});
  auto s_name = cudf::strings::concatenate(s_name_parts);

  // Generate the `s_nationkey` column
  auto s_nationkey = gen_rand_col<int>(0, 24, num_rows);

  // Generate the `s_phone` column
  auto s_phone_part_1 = cudf::strings::from_integers(gen_rand_col<int>(10, 34, num_rows)->view());
  auto s_phone_part_2 = cudf::strings::from_integers(gen_rand_col<int>(100, 999, num_rows)->view());
  auto s_phone_part_3 = cudf::strings::from_integers(gen_rand_col<int>(100, 999, num_rows)->view());
  auto s_phone_part_4 = cudf::strings::from_integers(gen_rand_col<int>(1000, 9999, num_rows)->view());
  auto s_phone_parts = cudf::table_view(
    {s_phone_part_1->view(), s_phone_part_2->view(), s_phone_part_3->view(), s_phone_part_4->view()});
  auto s_phone = cudf::strings::concatenate(s_phone_parts, cudf::string_scalar("-"));
  
  // Generate the `s_acctbal` column
  auto s_acctbal = gen_rand_col<float>(-999.99, 9999.99, num_rows);

  // Create the `supplier` table
  std::vector<std::unique_ptr<cudf::column>> columns;
  columns.push_back(std::move(s_suppkey));
  columns.push_back(std::move(s_name));
  columns.push_back(std::move(s_nationkey));
  columns.push_back(std::move(s_phone));
  columns.push_back(std::move(s_acctbal));
  
  return std::make_unique<cudf::table>(std::move(columns));
}

int main(int argc, char** argv)
{
  rmm::mr::cuda_memory_resource cuda_mr{};
  rmm::mr::pool_memory_resource mr{&cuda_mr, rmm::percent_of_free_device_memory(50)};
  rmm::mr::set_current_device_resource(&mr);

  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " <scale_factor>" << std::endl;
    return 1;
  }

  int32_t scale_factor = std::atoi(argv[1]);
  std::cout << "Requested scale factor: " << scale_factor << std::endl;

  auto supplier = generate_supplier(scale_factor);
  write_parquet(
    supplier->view(), "supplier.parquet", 
    {"s_suppkey", "s_name", "s_nationkey", "s_phone", "s_acctbal"}
  );

  auto nation = generate_nation(scale_factor);
  write_parquet(nation->view(), "nation.parquet", {"n_nationkey"});

  auto region = generate_region(scale_factor);
  write_parquet(region->view(), "region.parquet", {"r_regionkey"});

  return 0;
}
