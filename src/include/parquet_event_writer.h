#pragma once

#include "event_batch_storage.h"
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

class ParquetWriter
{
public:
  static constexpr bool enabled = true;
  static constexpr bool retains_batch = true;
  ParquetWriter(const std::string &path, const std::string &archive_id,
                int64_t generation, bool zstd, int column_workers);
  ~ParquetWriter();
  void write(EventBatch columns);
  void close();
  void abort();

private:
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<EventBatch> pending;
  bool eof{false};
  bool cancelled{false};
  bool finished{false};
  std::exception_ptr failure;
  int64_t rows{0};
  int64_t copied_rows{0};
  std::jthread worker;

  void run(const std::string &path, const std::string &archive_id,
           int64_t generation, bool zstd, int column_workers);
  void encode(const std::string &path, const std::string &archive_id,
              int64_t generation, bool zstd, int column_workers);
  EventBatch next();
};
