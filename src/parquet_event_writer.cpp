#include "parquet_event_writer.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <carquet/carquet.h>
#include <omp.h>

ParquetWriter::ParquetWriter(const std::string &path, const std::string &archive_id,
                             int64_t generation, bool zstd, int column_workers)
{
  if (column_workers < 1)
    throw std::invalid_argument("Parquet column workers must be positive");
  worker = std::jthread([this, path, archive_id, generation, zstd, column_workers]
  { run(path, archive_id, generation, zstd, column_workers); });
}

ParquetWriter::~ParquetWriter()
{
  abort();
}

void ParquetWriter::write(EventBatch columns)
{
  std::unique_lock lock(mutex);
  changed.wait(lock, [this]
  { return pending.size() < 2 || failure || finished; });
  if (failure)
    std::rethrow_exception(failure);
  if (finished)
    throw std::runtime_error("Parquet event writer stopped before accepting all batches");
  const auto count = static_cast<int64_t>(columns->event_indices.size());
  pending.push_back(std::move(columns));
  rows += count;
  changed.notify_all();
}

void ParquetWriter::close()
{
  {
    std::scoped_lock lock(mutex);
    eof = true;
    changed.notify_all();
  }
  if (worker.joinable())
    worker.join();
  if (failure)
    std::rethrow_exception(failure);
  if (copied_rows != rows)
    throw std::runtime_error("Parquet event writer produced the wrong row count");
}

void ParquetWriter::abort()
{
  if (!worker.joinable())
    return;
  {
    std::scoped_lock lock(mutex);
    cancelled = true;
    pending.clear();
    changed.notify_all();
  }
  worker.join();
}

void ParquetWriter::run(const std::string &path, const std::string &archive_id,
                        int64_t generation, bool zstd, int column_workers)
{
  try
  {
    encode(path, archive_id, generation, zstd, column_workers);
  }
  catch (...)
  {
    std::scoped_lock lock(mutex);
    failure = std::current_exception();
  }
  {
    std::scoped_lock lock(mutex);
    finished = true;
    changed.notify_all();
  }
}

EventBatch ParquetWriter::next()
{
  std::unique_lock lock(mutex);
  changed.wait(lock, [this]
  { return !pending.empty() || eof || cancelled; });
  if (pending.empty() || cancelled)
    return {};
  auto columns = std::move(pending.front());
  pending.pop_front();
  changed.notify_all();
  return columns;
}


namespace
{
void check_carquet(carquet_status_t status)
{
  if (status != CARQUET_OK)
    throw std::runtime_error(carquet_status_string(status));
}
} // namespace

void ParquetWriter::encode(const std::string &path, const std::string &archive_id,
                           int64_t generation, bool zstd, int column_workers)
{
  // OpenMP's setting belongs to this writer thread; simulation still uses TBB.
  omp_set_dynamic(0);
  omp_set_num_threads(column_workers);
  carquet_error_t error = CARQUET_ERROR_INIT;
  std::unique_ptr<carquet_schema_t, decltype(&carquet_schema_free)> schema(
    carquet_schema_create(&error), carquet_schema_free);
  if (!schema)
    throw std::runtime_error(error.message);
  for (size_t i = 0; i < event_column_names.size(); ++i)
    check_carquet(carquet_schema_add_column(schema.get(), event_column_names[i],
                                            i < 14 ? CARQUET_PHYSICAL_DOUBLE : CARQUET_PHYSICAL_INT64,
                                            nullptr, CARQUET_REPETITION_REQUIRED, 0, 0));
  carquet_writer_options_t options;
  carquet_writer_options_init(&options);
  options.compression = zstd ? CARQUET_COMPRESSION_ZSTD : CARQUET_COMPRESSION_UNCOMPRESSED;
  options.compression_level = 3;
  options.write_crc = false;
  options.row_group_size = std::numeric_limits<int64_t>::max();
  std::unique_ptr<carquet_writer_t, decltype(&carquet_writer_abort)> writer(
    carquet_writer_create(path.c_str(), schema.get(), &options, &error), carquet_writer_abort);
  if (!writer)
    throw std::runtime_error(error.message);
  check_carquet(carquet_writer_add_metadata(writer.get(), "apitofsim:event_schema", "1"));
  check_carquet(carquet_writer_add_metadata(writer.get(), "apitofsim:archive_id", archive_id.c_str()));
  check_carquet(carquet_writer_add_metadata(writer.get(), "apitofsim:generation", std::to_string(generation).c_str()));
  for (size_t i = 0; i < event_column_names.size(); ++i)
    check_carquet(carquet_writer_set_column_encoding(writer.get(), static_cast<int32_t>(i), CARQUET_ENCODING_RLE_DICTIONARY));
  size_t group_rows = 0;
  while (true)
  {
    auto columns = next();
    if (columns.get() == nullptr)
      break;
    const size_t count = columns->event_indices.size();
    auto validate = [count](const auto &values)
    {
      if (values.size() != count)
        throw std::invalid_argument("Inconsistent event column lengths");
    };
    for (const auto &values : columns->values)
      validate(values);
    for (const auto &values : columns->indices)
      validate(values);
    validate(columns->result_ids);
    for (size_t offset = 0; offset < count;)
    {
      const auto size = std::min(count - offset, 65536 - group_rows);
      int32_t column = 0;
      auto write = [&](const auto &values)
      {
        check_carquet(carquet_writer_write_batch(writer.get(), column++, values.data() + offset,
                                                 static_cast<int64_t>(size), nullptr, nullptr));
      };
      for (const auto &values : columns->values)
        write(values);
      for (const auto &values : columns->indices)
        write(values);
      write(columns->result_ids);
      write(columns->event_indices);
      offset += size;
      group_rows += size;
      copied_rows += static_cast<int64_t>(size);
      if (group_rows == 65536)
      {
        check_carquet(carquet_writer_new_row_group(writer.get()));
        group_rows = 0;
      }
    }
  }
  // close consumes the handle even on failure.
  check_carquet(carquet_writer_close(writer.release()));
}
