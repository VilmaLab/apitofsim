#pragma once

#include "event_columns.h"
#include <stdexcept>
#include <arrow/api.h>

inline void check_arrow(const arrow::Status &status)
{
  if (!status.ok())
    throw std::runtime_error(status.ToString());
}

template <typename T>
T unwrap_arrow(arrow::Result<T> &&result)
{
  check_arrow(result.status());
  return std::move(result).ValueUnsafe();
}

inline std::shared_ptr<arrow::Schema> event_schema()
{
  arrow::FieldVector fields;
  fields.reserve(event_column_names.size());
  for (size_t i = 0; i < event_column_names.size(); ++i)
    fields.push_back(arrow::field(event_column_names[i], i < 14 ? arrow::float64() : arrow::int64(), false));
  return arrow::schema(std::move(fields));
}

inline std::shared_ptr<arrow::RecordBatch> event_batch(
  const std::shared_ptr<arrow::Schema> &schema, const EventColumns &columns,
  const std::vector<int64_t> &results, const std::vector<int64_t> &indices,
  const std::shared_ptr<const EventColumns> &owner = {})
{
  const auto rows = static_cast<int64_t>(indices.size());
  arrow::ArrayVector arrays;
  arrays.reserve(event_column_names.size());
  auto add = [&](const auto &values)
  {
    if (values.size() != indices.size())
      throw std::invalid_argument("Inconsistent event column lengths");
    // IPC borrows the vectors for one synchronous write. Parquet's buffers
    // retain their shared owner even if the worker or writer keeps an array.
    auto buffer = arrow::Buffer::Wrap(values);
    if (owner)
      buffer = std::shared_ptr<arrow::Buffer>(buffer.get(), [buffer, owner](arrow::Buffer *) {});
    auto type = schema->field(static_cast<int>(arrays.size()))->type();
    arrays.push_back(arrow::MakeArray(arrow::ArrayData::Make(type, rows, {nullptr, std::move(buffer)}, 0)));
  };
  for (const auto &values : columns.values)
    add(values);
  for (const auto &values : columns.indices)
    add(values);
  add(results);
  add(indices);
  return arrow::RecordBatch::Make(schema, rows, std::move(arrays));
}
