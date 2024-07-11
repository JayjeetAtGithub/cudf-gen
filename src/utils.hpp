#include <vector>

#include <cudf/io/parquet.hpp>

void write_parquet(cudf::table_view tbl, std::string path,
                   std::vector<std::string> col_names) {
  std::cout << "Writing to " << path << "\n";
  auto sink_info = cudf::io::sink_info(path);
  cudf::io::table_metadata metadata;
  std::vector<cudf::io::column_name_info> col_name_infos;
  for (auto &col_name : col_names) {
    col_name_infos.push_back(cudf::io::column_name_info(col_name));
  }
  metadata.schema_info = col_name_infos;
  auto table_input_metadata = cudf::io::table_input_metadata{metadata};
  auto builder = cudf::io::parquet_writer_options::builder(sink_info, tbl);
  builder.metadata(table_input_metadata);
  auto options = builder.build();
  cudf::io::write_parquet(options);
}
