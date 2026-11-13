#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "event_batch_storage.h"

class IpcWriterBusy : public std::runtime_error
{
public:
  IpcWriterBusy() : std::runtime_error("Stop the active recording before using its IPC writer")
  {
  }
};

class EventWriter
{
  struct Impl;

public:
  // A run holds the lease through its last write and counter snapshot.
  class Lease
  {
  public:
    Lease(Lease &&) noexcept;
    ~Lease();
    void write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices);
    void write(const EventColumns &columns);
    void write(EventBatch columns);
    [[nodiscard]] bool is_parquet() const;
    [[nodiscard]] std::array<int64_t, 2> counts() const;

  private:
    friend class EventWriter;
    explicit Lease(std::shared_ptr<Impl> state);
    std::shared_ptr<Impl> impl;
  };

  EventWriter(const std::string &path, bool append, const std::string &archive_id);
  EventWriter(const std::string &path, const std::string &archive_id,
              int64_t generation, bool zstd, int column_workers);
  ~EventWriter();
  EventWriter(const EventWriter &) = delete;
  EventWriter &operator=(const EventWriter &) = delete;
  void write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices);
  [[nodiscard]] Lease acquire();
  void close(const std::optional<std::string> &metadata = std::nullopt);
  void abort();
  void set_metadata(const std::string &metadata);
  [[nodiscard]] int64_t batch_count() const;
  [[nodiscard]] int64_t rows_written() const;

private:
  std::shared_ptr<Impl> impl;
};

// Compatibility for callers of the original IPC-named recording facade.
using IpcEventWriter = EventWriter;
