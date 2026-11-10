#include "parquet_event_writer.h"
#include "arrow_event_batch.h"

#include <arrow/io/file.h>
#include <arrow/util/thread_pool.h>
#include <parquet/arrow/writer.h>
#include <parquet/metadata.h>

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

void ParquetEventWriter::run(const std::string &path, const std::string &archive_id,
                             int64_t generation, bool zstd, int column_workers)
{
  try
  {
    const auto schema = event_schema();
    auto file = unwrap_arrow(arrow::io::FileOutputStream::Open(path));
    parquet::WriterProperties::Builder properties;
    properties.max_row_group_length(65536);
    if (zstd)
      properties.compression(parquet::Compression::ZSTD)->compression_level(3);
    else
      properties.compression(parquet::Compression::UNCOMPRESSED);
    // Each writer has its own pool, independent of the simulation.
    auto executor = column_workers > 1 ? unwrap_arrow(arrow::internal::ThreadPool::Make(column_workers)) : nullptr;
    auto arrow_properties = parquet::ArrowWriterProperties::Builder()
                              .set_use_threads(executor != nullptr)
                              ->set_executor(executor.get())
                              ->build();
    auto writer = unwrap_arrow(parquet::arrow::FileWriter::Open(
      *schema, arrow::default_memory_pool(), file, properties.build(), arrow_properties));
    check_arrow(writer->AddKeyValueMetadata(arrow::key_value_metadata(
      {"apitofsim:event_schema", "apitofsim:archive_id", "apitofsim:generation"},
      {"1", archive_id, std::to_string(generation)})));
    while (auto columns = next())
      check_arrow(writer->WriteRecordBatch(*event_batch(schema, *columns, columns->result_ids, columns->event_indices, columns)));
    check_arrow(writer->Close());
    copied_rows = writer->metadata()->num_rows();
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

std::shared_ptr<EventColumns> ParquetEventWriter::next()
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
