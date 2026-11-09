#pragma once

#include "event_columns.h"
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

struct ArrowArray;
struct ArrowArrayStream;
struct ArrowSchema;

class ParquetEventWriter
{
public:
  ParquetEventWriter(const std::string &path, const std::string &archive_id,
                     int64_t generation, bool zstd, int column_workers);
  ~ParquetEventWriter();
  void write(const std::shared_ptr<EventColumns> &columns);
  void close();
  void abort();

private:
  std::mutex mutex;
  std::condition_variable changed;
  std::deque<std::shared_ptr<EventColumns>> pending;
  bool eof{false};
  bool cancelled{false};
  bool finished{false};
  std::exception_ptr failure;
  int64_t rows{0};
  int64_t copied_rows{0};
  std::jthread worker;

  void run(const std::string &path, const std::string &archive_id,
           int64_t generation, bool zstd, int column_workers);
  static int get_schema(ArrowArrayStream * /*unused*/, ArrowSchema *out);
  static int get_next(ArrowArrayStream *stream, ArrowArray *out);
  static const char *get_last_error(ArrowArrayStream * /*unused*/);
  static void release(ArrowArrayStream *stream);
};
