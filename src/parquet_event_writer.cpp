#include "parquet_event_writer.h"

#include <cerrno>
#include <nanoarrow/nanoarrow.hpp>
#include <stdexcept>
#include <utility>
#include <arrow/c/bridge.h>
#include <arrow/io/file.h>
#include <arrow/record_batch.h>
#include <arrow/util/key_value_metadata.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>

namespace
{
void check(int status)
{
  if (status != NANOARROW_OK)
    throw std::runtime_error("Could not export an Arrow event batch");
}

void make_schema(ArrowSchema *schema)
{
  ArrowSchemaInit(schema);
  check(ArrowSchemaSetTypeStruct(schema, event_column_names.size()));
  for (size_t i = 0; i < event_column_names.size(); ++i)
  {
    auto *field = schema->children[i];
    check(ArrowSchemaSetType(field, i < 14 ? NANOARROW_TYPE_DOUBLE : NANOARROW_TYPE_INT64));
    check(ArrowSchemaSetName(field, event_column_names[i]));
    field->flags = 0;
  }
}

struct Batch
{
  std::shared_ptr<EventColumns> columns;
  nanoarrow::UniqueArray array;
};

void check(const arrow::Status &status)
{
  if (!status.ok())
    throw std::runtime_error(status.ToString());
}

template <typename T>
T unwrap(arrow::Result<T> &&result)
{
  check(result.status());
  return std::move(result).ValueUnsafe();
}

int64_t write_parquet(const std::string &path, const std::string &archive_id,
                      int64_t generation, bool zstd, int column_workers, ArrowArrayStream &stream)
{
  auto reader = unwrap(arrow::ImportRecordBatchReader(&stream));
  auto file = unwrap(arrow::io::FileOutputStream::Open(path));
  parquet::WriterProperties::Builder properties;
  properties.max_row_group_length(65536);
  if (zstd)
    properties.compression(parquet::Compression::ZSTD)->compression_level(3);
  else
    properties.compression(parquet::Compression::UNCOMPRESSED);
  // Each writer has its own pool. The simulation and other Arrow users keep
  // their independent thread budgets.
  auto executor = column_workers > 1 ? unwrap(arrow::internal::ThreadPool::Make(column_workers)) : nullptr;
  auto arrow_properties = parquet::ArrowWriterProperties::Builder()
                            .set_use_threads(executor != nullptr)
                            ->set_executor(executor.get())
                            ->build();
  auto writer = unwrap(parquet::arrow::FileWriter::Open(
    *reader->schema(), arrow::default_memory_pool(), file, properties.build(), arrow_properties));
  check(writer->AddKeyValueMetadata(arrow::key_value_metadata(
    {"apitofsim:event_schema", "apitofsim:archive_id", "apitofsim:generation"},
    {"1", archive_id, std::to_string(generation)})));
  while (auto batch = unwrap(reader->Next()))
    check(writer->WriteRecordBatch(*batch));
  check(writer->Close());
  return writer->metadata()->num_rows();
}
} // namespace

ParquetEventWriter::ParquetEventWriter(const std::string &path, const std::string &archive_id,
                                       int64_t generation, bool zstd, int column_workers)
{
  if (column_workers < 1)
    throw std::invalid_argument("Parquet column workers must be positive");
  worker = std::jthread([this, path, archive_id, generation, zstd, column_workers]
  { run(path, archive_id, generation, zstd, column_workers); });
}

ParquetEventWriter::~ParquetEventWriter()
{
  abort();
}

void ParquetEventWriter::write(const std::shared_ptr<EventColumns> &columns)
{
  std::unique_lock lock(mutex);
  changed.wait(lock, [this]
  { return pending.size() < 2 || failure || finished; });
  if (failure)
    std::rethrow_exception(failure);
  if (finished)
    throw std::runtime_error("Parquet event writer stopped before accepting all batches");
  pending.push_back(columns);
  rows += static_cast<int64_t>(columns->event_indices.size());
  changed.notify_all();
}

void ParquetEventWriter::close()
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

void ParquetEventWriter::abort()
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

int ParquetEventWriter::get_schema(ArrowArrayStream * /*unused*/, ArrowSchema *out)
{
  try
  {
    nanoarrow::UniqueSchema schema;
    make_schema(schema.get());
    ArrowSchemaMove(schema.get(), out);
    return 0;
  }
  catch (...)
  {
    return EIO;
  }
}

int ParquetEventWriter::get_next(ArrowArrayStream *stream, ArrowArray *out)
{
  auto &writer = *static_cast<ParquetEventWriter *>(stream->private_data);
  try
  {
    std::shared_ptr<EventColumns> columns;
    {
      std::unique_lock lock(writer.mutex);
      writer.changed.wait(lock, [&writer]
      { return !writer.pending.empty() || writer.eof || writer.cancelled; });
      if (writer.pending.empty() || writer.cancelled)
      {
        out->release = nullptr;
        return 0;
      }
      columns = std::move(writer.pending.front());
      writer.pending.pop_front();
      writer.changed.notify_all();
    }
    auto batch = std::make_unique<Batch>();
    batch->columns = std::move(columns);
    nanoarrow::UniqueSchema schema;
    make_schema(schema.get());
    check(ArrowArrayInitFromSchema(batch->array.get(), schema.get(), nullptr));
    const auto &data = *batch->columns;
    const auto size = static_cast<int64_t>(data.event_indices.size());
    batch->array->length = size;
    batch->array->null_count = 0;
    auto set_column = [&](size_t i, const auto &column)
    {
      auto *child = batch->array->children[i];
      child->length = size;
      child->null_count = 0;
      child->buffers[1] = column.data();
    };
    for (size_t i = 0; i < data.values.size(); ++i)
      set_column(i, data.values[i]);
    for (size_t i = 0; i < data.indices.size(); ++i)
      set_column(data.values.size() + i, data.indices[i]);
    set_column(event_column_names.size() - 2, data.result_ids);
    set_column(event_column_names.size() - 1, data.event_indices);
    *out = *batch->array.get();
    out->private_data = batch.release();
    out->release = [](ArrowArray *array)
    {
      std::unique_ptr<Batch> owner(static_cast<Batch *>(array->private_data));
      array->release = nullptr;
    };
    return 0;
  }
  catch (...)
  {
    return EIO;
  }
}

const char *ParquetEventWriter::get_last_error(ArrowArrayStream * /*unused*/)
{
  return "Could not produce an Arrow event batch";
}

void ParquetEventWriter::release(ArrowArrayStream *stream)
{
  stream->release = nullptr;
}

void ParquetEventWriter::run(const std::string &path, const std::string &archive_id,
                             int64_t generation, bool zstd, int column_workers)
{
  try
  {
    ArrowArrayStream stream{get_schema, get_next, get_last_error, release, this};
    copied_rows = write_parquet(path, archive_id, generation, zstd, column_workers, stream);
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
