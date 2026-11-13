#pragma once

#include "event_columns.h"
#include <memory>
#include <string>

class ArrowWriter
{
  struct Impl;
  std::unique_ptr<Impl> impl;

public:
  static constexpr bool enabled = true;
  static constexpr bool retains_batch = false;
  ArrowWriter(const std::string &path, bool append, const std::string &archive_id);
  ~ArrowWriter();
  void write(const EventColumns &columns);
  void write(const EventColumns &columns, int64_t result_id, const std::vector<int64_t> &indices);
  void close();
  void abort();
  void set_metadata(const std::string &metadata);
  [[nodiscard]] int64_t batch_count() const;
};
