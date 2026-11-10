#include "event_recording.h"
#include "arrow_event_batch.h"
#include "arrow_ipc_footer.h"
#ifdef APITOFSIM_WITH_PARQUET
#include "parquet_event_writer.h"
#endif

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <arrow/io/memory.h>
#include <arrow/util/endian.h>

namespace
{
using arrow::ipc::internal::FileBlock;
constexpr std::array<char, 8> ipc_header{'A', 'R', 'R', 'O', 'W', '1', 0, 0};
constexpr std::array<unsigned char, 8> ipc_end{255, 255, 255, 255, 0, 0, 0, 0};
constexpr int64_t trailer_size = sizeof(int32_t) + 6;

// The stream must be seekable for append, where we overwrite the old ending.
class IpcOutputStream final : public arrow::io::OutputStream
{
public:
  mutable std::fstream file;

  arrow::Status Write(const void *data, int64_t length) override
  {
    file.write(static_cast<const char *>(data), length);
    return arrow::Status::OK();
  }

  arrow::Result<int64_t> Tell() const override
  {
    return static_cast<int64_t>(file.tellp());
  }

  arrow::Status Close() override
  {
    file.close();
    return arrow::Status::OK();
  }

  bool closed() const override
  {
    return !file.is_open();
  }
};

template <typename T>
T read_little_endian(const void *data)
{
  T value{};
  std::memcpy(&value, data, sizeof(value));
  return arrow::bit_util::FromLittleEndian(value);
}

// Arrow verifies the FlatBuffer before we reach this code. Its file reader
// exposes the schema and batch count but keeps the block descriptors private.
// Read just the two Block vectors from the verified Footer in Arrow's File.fbs.
const uint8_t *footer_blocks(const uint8_t *footer, uint16_t slot)
{
  const auto *table = footer + read_little_endian<uint32_t>(footer);
  const auto *vtable = table - read_little_endian<int32_t>(table);
  if (slot >= read_little_endian<uint16_t>(vtable))
    return nullptr;
  const auto field = read_little_endian<uint16_t>(vtable + slot);
  if (field == 0)
    return nullptr;
  const auto *offset = table + field;
  return offset + read_little_endian<uint32_t>(offset);
}

std::vector<FileBlock> read_blocks(const uint8_t *footer, int64_t end)
{
  constexpr uint16_t dictionaries_slot = 8;
  constexpr uint16_t record_batches_slot = 10;
  constexpr size_t block_size = 24;
  const auto *dictionaries = footer_blocks(footer, dictionaries_slot);
  if (dictionaries != nullptr && read_little_endian<uint32_t>(dictionaries) != 0)
    throw std::runtime_error("Event IPC: dictionary batches are unsupported");

  const auto *batches = footer_blocks(footer, record_batches_slot);
  if (batches == nullptr)
    return {};
  const auto count = read_little_endian<uint32_t>(batches);
  std::vector<FileBlock> blocks;
  blocks.reserve(count);
  for (uint32_t i = 0; i < count; ++i)
  {
    const auto *data = batches + sizeof(uint32_t) + (i * block_size);
    const FileBlock block{read_little_endian<int64_t>(data), read_little_endian<int32_t>(data + 8),
                          read_little_endian<int64_t>(data + 16)};
    if (block.offset < 8 || block.metadata_length <= 0 || block.body_length < 0 ||
        block.offset > end - block.metadata_length || block.body_length > end - block.offset - block.metadata_length)
      throw std::runtime_error("Event IPC: invalid batch offsets");
    blocks.push_back(block);
  }
  return blocks;
}
} // namespace

struct IpcEventWriter::Impl
{
  std::mutex mutex;
  bool leased{false};
  bool sealed{false};
  std::filesystem::path path;
  IpcOutputStream output;
  std::shared_ptr<arrow::Schema> schema;
  arrow::ipc::IpcWriteOptions options = arrow::ipc::IpcWriteOptions::Defaults();
  std::vector<FileBlock> blocks;
  int64_t rows{0};
  bool closed{false};
  bool write_failed{false};
  std::string executions{"{}"};
#ifdef APITOFSIM_WITH_PARQUET
  std::unique_ptr<ParquetEventWriter> parquet;

  Impl(const std::string &path, const std::string &archive_id, int64_t generation, bool zstd, int column_workers)
      : parquet(std::make_unique<ParquetEventWriter>(
          path, archive_id, generation, zstd, column_workers))
  {
  }
#endif

  void require_idle() const
  {
    if (leased)
      throw IpcWriterBusy();
  }

  std::array<int64_t, 2> counts() const
  {
    return {static_cast<int64_t>(blocks.size()), rows};
  }

  Impl(const std::string &path, bool append, const std::string &archive_id)
      : path(std::u8string(path.begin(), path.end())),
        schema(event_schema()->WithMetadata(arrow::key_value_metadata(
          {"apitofsim:event_schema", "apitofsim:archive_id"}, {"1", archive_id})))
  {
    auto &file = output.file;
    file.exceptions(std::ios::failbit | std::ios::badbit);
    if (append)
    {
      file.open(this->path, std::ios::binary | std::ios::in | std::ios::out);
      reopen(archive_id);
    }
    else
    {
      file.open(this->path, std::ios::binary | std::ios::out | std::ios::trunc);
      check_arrow(output.Write(ipc_header.data(), ipc_header.size()));
      const auto message = unwrap_arrow(arrow::ipc::SerializeSchema(*schema));
      check_arrow(output.Write(message->data(), message->size()));
    }
  }

  void reopen(const std::string &archive_id)
  {
    auto &file = output.file;
    file.seekg(0, std::ios::end);
    const auto size = static_cast<int64_t>(file.tellg());
    constexpr int64_t header_size = ipc_header.size();
    constexpr int64_t ending_size = trailer_size + ipc_end.size();
    if (size < ending_size + header_size)
      throw std::runtime_error("Event IPC: truncated archive");
    std::array<char, ipc_header.size()> header{};
    file.seekg(0);
    file.read(header.data(), header.size());
    if (header != ipc_header)
      throw std::runtime_error("Event IPC: invalid archive header");
    std::array<char, trailer_size> trailer{};
    file.seekg(size - trailer_size);
    file.read(trailer.data(), trailer.size());
    const auto footer_size = read_little_endian<int32_t>(trailer.data());
    if (footer_size <= 0 || footer_size > size - ending_size - header_size)
      throw std::runtime_error("Event IPC: invalid archive footer");
    const auto end = size - ending_size - footer_size;

    std::vector<char> tail(static_cast<size_t>(footer_size + ending_size));
    file.seekg(end);
    file.read(tail.data(), static_cast<std::streamsize>(tail.size()));
    if (std::memcmp(tail.data(), ipc_end.data(), ipc_end.size()) != 0)
      throw std::runtime_error("Event IPC: invalid end marker");

    // Opening a file reader only reads the trailer and footer. Give it the
    // same bytes we will use for the block offsets, with no batch data at all.
    auto buffer = arrow::Buffer::Wrap(tail);
    auto reader = unwrap_arrow(arrow::ipc::RecordBatchFileReader::Open(
      std::make_shared<arrow::io::BufferReader>(buffer)));
    const auto &old_schema = reader->schema();
    if (old_schema->num_fields() != schema->num_fields())
      throw std::runtime_error("Event IPC: unsupported event schema");
    for (int i = 0; i < schema->num_fields(); ++i)
    {
      const auto &field = old_schema->field(i);
      const auto &expected = schema->field(i);
      if (field->name() != expected->name() || !field->type()->Equals(expected->type()))
        throw std::runtime_error("Event IPC: incompatible event schema");
    }
    const auto &metadata = old_schema->metadata();
    const auto identity = metadata ? metadata->Get("apitofsim:archive_id") : arrow::Status::KeyError("Missing archive identity");
    if (!identity.ok() || *identity != archive_id)
      throw std::runtime_error("Event IPC: archive identity mismatch");
    blocks = read_blocks(buffer->data() + ipc_end.size(), end);
    file.seekp(end);
  }

  void write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &indices)
  {
#ifdef APITOFSIM_WITH_PARQUET
    if (parquet)
      throw std::invalid_argument("Analytical event writing requires an owned batch");
#endif
    if (closed)
      throw std::runtime_error("Event IPC: writer is closed");
    if (indices.empty())
      return;
    const std::vector<int64_t> results(indices.size(), result_id);
    const auto batch = event_batch(schema, columns, results, indices);
    FileBlock block{unwrap_arrow(output.Tell()), 0, 0};
    check_arrow(arrow::ipc::WriteRecordBatch(*batch, 0, &output, &block.metadata_length, &block.body_length, options));
    blocks.push_back(block);
    rows += batch->num_rows();
  }

  void close()
  {
    if (write_failed)
      throw std::runtime_error("Event IPC: cannot publish after a failed write");
    if (closed)
      return;
#ifdef APITOFSIM_WITH_PARQUET
    if (parquet)
    {
      parquet->close();
      closed = true;
      return;
    }
#endif
    check_arrow(output.Write(ipc_end.data(), ipc_end.size()));
    const auto start = unwrap_arrow(output.Tell());
    check_arrow(arrow::ipc::internal::WriteFileFooter(
      *schema, {}, blocks, arrow::key_value_metadata({"apitofsim:executions"}, {executions}), &output));
    const auto size = unwrap_arrow(output.Tell()) - start;
    if (size <= 0 || size > std::numeric_limits<int32_t>::max())
      throw std::overflow_error("Event IPC: invalid footer size");
    const auto length = arrow::bit_util::ToLittleEndian(static_cast<int32_t>(size));
    check_arrow(output.Write(&length, sizeof(length)));
    check_arrow(output.Write(ipc_header.data(), 6));
    const auto end = static_cast<uintmax_t>(unwrap_arrow(output.Tell()));
    check_arrow(output.Close());
    std::filesystem::resize_file(path, end);
    closed = true;
  }
};

IpcEventWriter::IpcEventWriter(const std::string &path, bool append, const std::string &archive_id) : impl(std::make_shared<Impl>(path, append, archive_id))
{
}
IpcEventWriter::IpcEventWriter(const std::string &path, const std::string &archive_id,
                               int64_t generation, bool zstd, int column_workers)
#ifdef APITOFSIM_WITH_PARQUET
    : impl(std::make_shared<Impl>(path, archive_id, generation, zstd, column_workers))
{
}
#else
{
  (void)path;
  (void)archive_id;
  (void)generation;
  (void)zstd;
  (void)column_workers;
  throw std::runtime_error("Parquet recording requires a build with -Dnative_parquet_writer=enabled");
}
#endif
IpcEventWriter::~IpcEventWriter() = default;

IpcEventWriter::Lease::Lease(std::shared_ptr<Impl> state) : impl(std::move(state))
{
}
IpcEventWriter::Lease::Lease(Lease &&) noexcept = default;
IpcEventWriter::Lease::~Lease()
{
  if (impl)
  {
    std::scoped_lock lock(impl->mutex);
    impl->leased = false;
  }
}

IpcEventWriter::Lease IpcEventWriter::acquire()
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  if (impl->sealed || impl->write_failed)
    throw std::runtime_error("Event IPC: writer is closed or has a failed write");
  impl->leased = true;
  return Lease(impl);
}

void IpcEventWriter::Lease::write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices)
{
  try
  {
    impl->write(columns, result_id, event_indices);
  }
  catch (const std::invalid_argument &)
  {
    // Shape validation happens before encoding or writing anything.
    throw;
  }
  catch (...)
  {
    if (!impl->closed)
      impl->write_failed = true;
    throw;
  }
}

bool IpcEventWriter::Lease::write(const std::shared_ptr<EventColumns> &columns, int64_t result_id)
{
#ifdef APITOFSIM_WITH_PARQUET
  if (impl->parquet)
  {
    try
    {
      columns->result_ids.assign(columns->event_indices.size(), result_id);
      impl->parquet->write(columns);
      impl->rows += static_cast<int64_t>(columns->event_indices.size());
    }
    catch (...)
    {
      impl->write_failed = true;
      throw;
    }
    return true;
  }
#endif
  write(*columns, result_id, columns->event_indices);
  return false;
}

std::array<int64_t, 2> IpcEventWriter::Lease::counts() const
{
  return impl->counts();
}

void IpcEventWriter::write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices)
{
  acquire().write(columns, result_id, event_indices);
}

void IpcEventWriter::close(const std::optional<std::string> &metadata)
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  if (impl->sealed && !impl->closed)
    throw std::runtime_error("Event IPC: previous close failed; abort the writer");
  if (metadata)
    impl->executions = *metadata;
  impl->sealed = true;
  impl->close();
}
int64_t IpcEventWriter::batch_count() const
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  return impl->counts()[0];
}
int64_t IpcEventWriter::rows_written() const
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  return impl->counts()[1];
}

void IpcEventWriter::set_metadata(const std::string &metadata)
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  impl->executions = metadata;
}
void IpcEventWriter::abort()
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  impl->sealed = true;
#ifdef APITOFSIM_WITH_PARQUET
  if (impl->parquet)
    impl->parquet->abort();
#endif
  impl->output.file.exceptions(std::ios::goodbit);
  if (impl->output.file.is_open())
    impl->output.file.close();
  impl->closed = true;
}
