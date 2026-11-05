#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "event_columns.h"

// One consuming thread owns a writer. The workflow owns locking, journaling,
// fsync and the transaction that publishes its resulting file generation.
class IpcEventWriter
{
public:
  IpcEventWriter(const std::string &path, bool append, const std::string &archive_id);
  ~IpcEventWriter();
  IpcEventWriter(const IpcEventWriter &) = delete;
  IpcEventWriter &operator=(const IpcEventWriter &) = delete;
  void write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &event_indices);
  void close();
  void abort();
  void set_metadata(const std::string &metadata);
  [[nodiscard]] int64_t batch_count() const;
  [[nodiscard]] int64_t rows_written() const;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;
};
