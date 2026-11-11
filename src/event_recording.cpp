#include "event_recording.h"
#include "parquet_event_writer.h"

#include <array>
#include <cstring>
#include <cstdlib>
#include <flatcc/flatcc_builder.h>
#include <nanoarrow/ipc/flatcc_generated.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <nanoarrow/nanoarrow_ipc.hpp>

namespace
{
constexpr auto &names = event_column_names;

void check(int status, const ArrowError &error = {})
{
  if (status != NANOARROW_OK)
    throw std::runtime_error("Event IPC: " + std::string(error.message) + " (error " + std::to_string(status) + ")");
}

ArrowBufferView view(const void *data, int64_t size)
{
  ArrowBufferView out{};
  out.data.data = data; // NOLINT(cppcoreguidelines-pro-type-union-access): nanoarrow buffer view API.
  out.size_bytes = size;
  return out;
}

std::array<char, 4> little_endian(uint32_t value)
{
  return {static_cast<char>(value), static_cast<char>(value >> 8U), static_cast<char>(value >> 16U), static_cast<char>(value >> 24U)};
}
} // namespace

struct IpcEventWriter::Impl
{
  std::mutex mutex;
  bool leased{false};
  bool sealed{false};
  std::filesystem::path path;
  std::fstream file;
  nanoarrow::ipc::UniqueEncoder encoder;
  nanoarrow::ipc::UniqueFooter footer;
  nanoarrow::UniqueArrayView array;
  nanoarrow::UniqueArray data;
  nanoarrow::UniqueBuffer metadata;
  nanoarrow::UniqueBuffer body;
  int64_t rows{0};
  bool closed{false};
  bool write_failed{false};
  std::string executions{"{}"};
  std::unique_ptr<ParquetEventWriter> parquet;

  Impl(const std::string &path, const std::string &archive_id, int64_t generation, bool zstd, int column_workers)
      : parquet(std::make_unique<ParquetEventWriter>(
          path, archive_id, generation, zstd, column_workers))
  {
  }

  void require_idle() const
  {
    if (leased)
      throw IpcWriterBusy();
  }

  std::array<int64_t, 2> counts() const
  {
    if (parquet)
      return {0, rows};
    return {footer->record_batch_blocks.size_bytes / static_cast<int64_t>(sizeof(ArrowIpcFileBlock)), rows};
  }

  Impl(const std::string &path, bool append, const std::string &archive_id) : path(std::u8string(path.begin(), path.end()))
  {
    ArrowError error{};
    check(ArrowIpcEncoderInit(encoder.get()));
    ArrowSchemaInit(&footer->schema);
    check(ArrowSchemaSetTypeStruct(&footer->schema, names.size()));
    for (size_t i = 0; i < names.size(); ++i)
    {
      auto *field = footer->schema.children[i];
      check(ArrowSchemaSetType(field, i < 14 ? NANOARROW_TYPE_DOUBLE : NANOARROW_TYPE_INT64));
      check(ArrowSchemaSetName(field, names[i]));
      field->flags = 0;
    }
    nanoarrow::UniqueBuffer schema_metadata;
    check(ArrowMetadataBuilderInit(schema_metadata.get(), nullptr));
    check(ArrowMetadataBuilderAppend(schema_metadata.get(), ArrowCharView("apitofsim:event_schema"), ArrowCharView("1")));
    check(ArrowMetadataBuilderAppend(schema_metadata.get(), ArrowCharView("apitofsim:archive_id"), ArrowCharView(archive_id.c_str())));
    check(ArrowSchemaSetMetadata(&footer->schema, reinterpret_cast<const char *>(schema_metadata->data))); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): binary IPC buffer.
    file.exceptions(std::ios::failbit | std::ios::badbit);
    if (!append)
    {
      std::ofstream create(this->path, std::ios::binary | std::ios::trunc);
      create.exceptions(std::ios::failbit | std::ios::badbit);
      create.close();
    }
    file.open(this->path, std::ios::binary | std::ios::in | std::ios::out);
    if (append)
    {
      reopen(error, archive_id);
    }
    else
    {
      file.write("ARROW1\0\0", 8);
      check(ArrowIpcEncoderEncodeSchema(encoder.get(), &footer->schema, &error), error);
      check(ArrowIpcEncoderFinalizeBuffer(encoder.get(), 1, metadata.get()));
      write_buffer(metadata.get());
    }
    check(ArrowArrayViewInitFromSchema(array.get(), &footer->schema, &error), error);
    check(ArrowArrayInitFromSchema(data.get(), &footer->schema, &error), error);
  }

  void reopen(ArrowError &error, const std::string &archive_id)
  {
    file.seekg(0, std::ios::end);
    const auto size = static_cast<int64_t>(file.tellg());
    if (size < 26)
      throw std::runtime_error("Event IPC: truncated archive");
    std::array<char, 8> header{};
    file.seekg(0);
    file.read(header.data(), header.size());
    if (std::memcmp(header.data(), "ARROW1\0\0", header.size()) != 0)
      throw std::runtime_error("Event IPC: invalid archive header");
    std::array<unsigned char, 10> ending{};
    file.seekg(size - 10);
    file.read(reinterpret_cast<char *>(ending.data()), ending.size()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): binary IPC buffer.
    const auto footer_size = static_cast<int64_t>(static_cast<uint32_t>(ending[0]) | (static_cast<uint32_t>(ending[1]) << 8U) |
                                                  (static_cast<uint32_t>(ending[2]) << 16U) | (static_cast<uint32_t>(ending[3]) << 24U));
    if (std::memcmp(ending.data() + 4, "ARROW1", 6) != 0 || footer_size > size - 26 || footer_size <= 0)
      throw std::runtime_error("Event IPC: invalid archive footer");
    std::vector<char> tail(static_cast<size_t>(footer_size + 10));
    file.seekg(size - static_cast<int64_t>(tail.size()));
    file.read(tail.data(), static_cast<std::streamsize>(tail.size()));
    nanoarrow::ipc::UniqueDecoder decoder;
    check(ArrowIpcDecoderInit(decoder.get()));
    auto data = view(tail.data(), static_cast<int64_t>(tail.size()));
    check(ArrowIpcDecoderVerifyFooter(decoder.get(), data, &error), error);
    check(ArrowIpcDecoderDecodeFooter(decoder.get(), data, &error), error);
    const auto *old = decoder->footer;
    if (std::cmp_not_equal(old->schema.n_children, names.size()) || old->dictionary_blocks.size_bytes != 0)
      throw std::runtime_error("Event IPC: unsupported event schema");
    for (size_t i = 0; i < names.size(); ++i)
    {
      const auto *field = old->schema.children[i];
      if (field->name == nullptr || field->format == nullptr || field->dictionary != nullptr || field->n_children != 0 ||
          std::string_view(field->name) != names[i] || std::strcmp(field->format, footer->schema.children[i]->format) != 0)
        throw std::runtime_error("Event IPC: incompatible event schema");
    }
    ArrowStringView id{};
    check(ArrowMetadataGetValue(old->schema.metadata, ArrowCharView("apitofsim:archive_id"), &id));
    if (id.data == nullptr || std::string_view(id.data, static_cast<size_t>(id.size_bytes)) != archive_id)
      throw std::runtime_error("Event IPC: archive identity mismatch");
    check(ArrowBufferAppend(&footer->record_batch_blocks, old->record_batch_blocks.data, old->record_batch_blocks.size_bytes));
    // Our files always end their embedded stream with the eight-byte EOS marker.
    const int64_t offset = size - footer_size - 18;
    const auto count = footer->record_batch_blocks.size_bytes / static_cast<int64_t>(sizeof(ArrowIpcFileBlock));
    for (int64_t i = 0; i < count; ++i)
    {
      ArrowIpcFileBlock block{};
      std::memcpy(&block, footer->record_batch_blocks.data + (i * sizeof(block)), sizeof(block));
      if (block.offset < 8 || block.metadata_length <= 0 || block.body_length < 0 ||
          block.offset > offset - block.metadata_length || block.body_length > offset - block.offset - block.metadata_length)
        throw std::runtime_error("Event IPC: invalid batch offsets");
    }
    std::array<unsigned char, 8> eos{};
    file.seekg(offset);
    file.read(reinterpret_cast<char *>(eos.data()), eos.size()); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): binary IPC buffer.
    constexpr std::array<unsigned char, 8> expected_eos{255, 255, 255, 255, 0, 0, 0, 0};
    if (eos != expected_eos)
      throw std::runtime_error("Event IPC: invalid end marker");
    file.seekp(offset);
  }

  void write_buffer(const ArrowBuffer *buffer)
  {
    file.write(reinterpret_cast<const char *>(buffer->data), buffer->size_bytes); // NOLINT(cppcoreguidelines-pro-type-reinterpret-cast): binary IPC buffer.
  }

  void write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &indices)
  {
    if (parquet)
      throw std::invalid_argument("Analytical event writing requires an owned batch");
    if (closed)
      throw std::runtime_error("Event IPC: writer is closed");
    const auto count = static_cast<int64_t>(indices.size());
    if (count == 0)
      return;
    std::vector<int64_t> results(indices.size(), result_id);
    data->length = count;
    data->null_count = 0;
    auto column = [&](size_t i, const auto &values)
    {
      if (values.size() != indices.size())
        throw std::invalid_argument("Event IPC: inconsistent column lengths");
      auto *child = data->children[i];
      child->length = count;
      child->null_count = 0;
      child->buffers[0] = nullptr;
      child->buffers[1] = values.data();
    };
    for (size_t i = 0; i < columns.values.size(); ++i)
      column(i, columns.values[i]);
    for (size_t i = 0; i < columns.indices.size(); ++i)
      column(i + columns.values.size(), columns.indices[i]);
    column(20, results);
    column(21, indices);
    check(ArrowBufferResize(metadata.get(), 0, 0));
    check(ArrowBufferResize(body.get(), 0, 0));
    ArrowError error{};
    check(ArrowArrayViewSetArray(array.get(), data.get(), &error), error);
    check(ArrowIpcEncoderEncodeSimpleRecordBatch(encoder.get(), array.get(), body.get(), &error), error);
    check(ArrowIpcEncoderFinalizeBuffer(encoder.get(), 1, metadata.get()));
    if (metadata->size_bytes > std::numeric_limits<int32_t>::max())
      throw std::overflow_error("Event IPC: batch metadata too large");
    const ArrowIpcFileBlock block{static_cast<int64_t>(file.tellp()), static_cast<int32_t>(metadata->size_bytes), body->size_bytes};
    check(ArrowBufferAppend(&footer->record_batch_blocks, &block, sizeof(block)));
    write_buffer(metadata.get());
    write_buffer(body.get());
    rows += count;
  }

  void close()
  {
    if (write_failed)
      throw std::runtime_error("Event IPC: cannot publish after a failed write");
    if (closed)
      return;
    if (parquet)
    {
      parquet->close();
      closed = true;
      return;
    }
    const auto continuation = little_endian(UINT32_MAX);
    const auto zero = little_endian(0);
    file.write(continuation.data(), continuation.size());
    file.write(zero.data(), zero.size());
    check(ArrowBufferResize(metadata.get(), 0, 0));
    ArrowError error{};
    check(ArrowIpcEncoderEncodeFooter(encoder.get(), footer.get(), &error), error);
    check(ArrowIpcEncoderFinalizeBuffer(encoder.get(), 0, metadata.get()));
    // The stable schema remains identical in the initial message and footer.
    // Per-execution completion records belong to the footer's custom metadata.
    struct Builder
    {
      flatcc_builder_t value{};
      Builder()
      {
        check(flatcc_builder_init(&value));
      }
      ~Builder()
      {
        flatcc_builder_clear(&value);
      }
    } builder;
    auto *b = &builder.value;
    const auto *original = org_apache_arrow_flatbuf_Footer_as_root(metadata->data);
    check(org_apache_arrow_flatbuf_Footer_start_as_root(b));
    check(org_apache_arrow_flatbuf_Footer_version_pick(b, original));
    check(org_apache_arrow_flatbuf_Footer_schema_pick(b, original));
    check(org_apache_arrow_flatbuf_Footer_dictionaries_pick(b, original));
    check(org_apache_arrow_flatbuf_Footer_recordBatches_pick(b, original));
    check(org_apache_arrow_flatbuf_Footer_custom_metadata_start(b));
    check(org_apache_arrow_flatbuf_Footer_custom_metadata_push_start(b));
    check(org_apache_arrow_flatbuf_KeyValue_key_create_str(b, "apitofsim:executions"));
    check(org_apache_arrow_flatbuf_KeyValue_value_create_str(b, executions.c_str()));
    if (org_apache_arrow_flatbuf_Footer_custom_metadata_push_end(b) == nullptr)
      throw std::bad_alloc();
    check(org_apache_arrow_flatbuf_Footer_custom_metadata_end(b));
    if (org_apache_arrow_flatbuf_Footer_end_as_root(b) == 0)
      throw std::bad_alloc();
    size_t footer_size = 0;
    std::unique_ptr<void, decltype(&std::free)> encoded(flatcc_builder_finalize_buffer(b, &footer_size), &std::free);
    if (!encoded)
      throw std::bad_alloc();
    check(ArrowBufferResize(metadata.get(), 0, 0));
    check(ArrowBufferAppend(metadata.get(), encoded.get(), static_cast<int64_t>(footer_size)));
    if (metadata->size_bytes > std::numeric_limits<int32_t>::max())
      throw std::overflow_error("Event IPC: footer too large");
    write_buffer(metadata.get());
    const auto size = little_endian(static_cast<uint32_t>(metadata->size_bytes));
    file.write(size.data(), size.size());
    file.write("ARROW1", 6);
    const auto end = static_cast<uintmax_t>(file.tellp());
    file.flush();
    file.close();
    std::filesystem::resize_file(path, end);
    closed = true;
  }
};

IpcEventWriter::IpcEventWriter(const std::string &path, bool append, const std::string &archive_id) : impl(std::make_shared<Impl>(path, append, archive_id))
{
}
IpcEventWriter::IpcEventWriter(const std::string &path, const std::string &archive_id,
                               int64_t generation, bool zstd, int column_workers)
    : impl(std::make_shared<Impl>(path, archive_id, generation, zstd, column_workers))
{
}
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
  if (impl->parquet)
    impl->parquet->abort();
  impl->file.exceptions(std::ios::goodbit);
  if (impl->file.is_open())
    impl->file.close();
  impl->closed = true;
}
