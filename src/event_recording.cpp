#include "event_recording.h"
#include "arrow_event_writer.h"
#include "parquet_event_writer.h"
#include <mutex>
#include <variant>

struct EventWriter::Impl
{
  std::mutex mutex;
  bool leased{false};
  bool sealed{false};
  bool closed{false};
  bool write_failed{false};
  int64_t rows{0};
  std::variant<std::unique_ptr<ArrowWriter>, std::unique_ptr<ParquetWriter>> backend;

  Impl(const std::string &path, bool append, const std::string &archive_id)
      : backend(std::make_unique<ArrowWriter>(path, append, archive_id))
  {
  }
  Impl(const std::string &path, const std::string &archive_id, int64_t generation, bool zstd, int column_workers)
      : backend(std::make_unique<ParquetWriter>(path, archive_id, generation, zstd, column_workers))
  {
  }

  void require_idle() const
  {
    if (leased)
      throw IpcWriterBusy();
  }
  ArrowWriter &arrow() const
  {
    return *std::get<std::unique_ptr<ArrowWriter>>(backend);
  }
  ParquetWriter &parquet() const
  {
    return *std::get<std::unique_ptr<ParquetWriter>>(backend);
  }
  bool is_parquet() const
  {
    return std::holds_alternative<std::unique_ptr<ParquetWriter>>(backend);
  }
  std::array<int64_t, 2> counts() const
  {
    return {is_parquet() ? 0 : arrow().batch_count(), rows};
  }

  template <typename Encode>
  void write(const Encode &encode, int64_t count)
  {
    if (closed || write_failed)
      throw std::runtime_error("Event writer is closed or has a failed write");
    try
    {
      encode();
      rows += count;
    }
    catch (const std::invalid_argument &)
    {
      // Shape validation precedes encoding, so the caller can correct a batch.
      throw;
    }
    catch (...)
    {
      write_failed = true;
      throw;
    }
  }
  void close()
  {
    if (write_failed)
      throw std::runtime_error("Event writer cannot publish after a failed write");
    if (closed)
      return;
    std::visit([](auto &writer)
    { writer->close(); }, backend);
    closed = true;
  }
  void set_metadata(const std::string &metadata) const
  {
    if (!is_parquet())
      arrow().set_metadata(metadata);
  }
};

EventWriter::EventWriter(const std::string &path, bool append, const std::string &archive_id)
    : impl(std::make_shared<Impl>(path, append, archive_id))
{
}
EventWriter::EventWriter(const std::string &path, const std::string &archive_id, int64_t generation, bool zstd, int column_workers)
    : impl(std::make_shared<Impl>(path, archive_id, generation, zstd, column_workers))
{
}
EventWriter::~EventWriter() = default;
EventWriter::Lease::Lease(std::shared_ptr<Impl> state) : impl(std::move(state))
{
}
EventWriter::Lease::Lease(Lease &&) noexcept = default;
EventWriter::Lease::~Lease()
{
  if (impl)
  {
    std::scoped_lock lock(impl->mutex);
    impl->leased = false;
  }
}
EventWriter::Lease EventWriter::acquire()
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  if (impl->sealed || impl->write_failed)
    throw std::runtime_error("Event writer is closed or has a failed write");
  impl->leased = true;
  return Lease(impl);
}
void EventWriter::Lease::write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices)
{
  impl->write([&]
  { impl->arrow().write(columns, result_id, event_indices); }, static_cast<int64_t>(event_indices.size()));
}
void EventWriter::Lease::write(const EventColumns &columns)
{
  impl->write([&]
  { impl->arrow().write(columns); }, static_cast<int64_t>(columns.event_indices.size()));
}
void EventWriter::Lease::write(EventBatch columns)
{
  const auto count = static_cast<int64_t>(columns->event_indices.size());
  impl->write([&]
  { impl->parquet().write(std::move(columns)); }, count);
}
bool EventWriter::Lease::is_parquet() const
{
  return impl->is_parquet();
}
std::array<int64_t, 2> EventWriter::Lease::counts() const
{
  return impl->counts();
}
void EventWriter::write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices)
{
  acquire().write(columns, result_id, event_indices);
}
void EventWriter::close(const std::optional<std::string> &metadata)
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  if (impl->sealed && !impl->closed)
    throw std::runtime_error("Event writer previous close failed; abort the writer");
  if (metadata)
    impl->set_metadata(*metadata);
  impl->sealed = true;
  impl->close();
}
int64_t EventWriter::batch_count() const
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  return impl->counts()[0];
}
int64_t EventWriter::rows_written() const
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  return impl->counts()[1];
}
void EventWriter::set_metadata(const std::string &metadata)
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  impl->set_metadata(metadata);
}
void EventWriter::abort()
{
  std::scoped_lock lock(impl->mutex);
  impl->require_idle();
  impl->sealed = true;
  std::visit([](auto &writer)
  { writer->abort(); }, impl->backend);
  impl->closed = true;
}
