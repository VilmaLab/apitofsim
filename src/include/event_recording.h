#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>
#include "event_columns.h"

class IpcWriterBusy : public std::runtime_error
{
public:
  IpcWriterBusy() : std::runtime_error("Stop the active recording before using its IPC writer")
  {
  }
};

class IpcEventWriter
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
    // True when the writer takes shared ownership of the batch.
    bool write(const std::shared_ptr<EventColumns> &columns, int64_t result_id);
    [[nodiscard]] std::array<int64_t, 2> counts() const;

  private:
    friend class IpcEventWriter;
    explicit Lease(std::shared_ptr<Impl> state);
    std::shared_ptr<Impl> impl;
  };

  IpcEventWriter(const std::string &path, bool append, const std::string &archive_id);
  IpcEventWriter(const std::string &path, const std::string &archive_id,
                 int64_t generation, bool zstd, int column_workers);
  ~IpcEventWriter();
  IpcEventWriter(const IpcEventWriter &) = delete;
  IpcEventWriter &operator=(const IpcEventWriter &) = delete;
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
