#pragma once

#include "utils.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cstring>
#include <fstream>
#include <Eigen/Dense>
#include <magic_enum/magic_enum.hpp>
#pragma clang attribute push(__attribute__((no_sanitize("unsigned-integer-overflow"))), apply_to = function)
#include <blockingconcurrentqueue.h>
#pragma clang attribute pop

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_thread_num() 0
#endif

using magic_enum::enum_count;
using moodycamel::BlockingConcurrentQueue;

template <typename Callback>
std::string call_with_stringstream(Callback cb)
{
  std::stringstream ss;
  ss << std::scientific << std::setprecision(3);
  cb(ss);
  return ss.str();
}

class ApiTofError : public std::runtime_error
{
public:
  ApiTofError(const std::string &msg)
      : std::runtime_error(msg)
  {
  }

  ApiTofError(const char *msg)
      : std::runtime_error(msg)
  {
  }

  template <typename Callback>
  ApiTofError(Callback cb)
      : ApiTofError(call_with_stringstream(cb))
  {
  }
};

namespace Counter
{
enum Counter
{
  nwarnings,
  n_fragmented_total,
  n_escaped_total,
  ncoll_total,
  counter_collision_rejections
};
};
constexpr auto n_counters = enum_count<Counter::Counter>();
using Counters = Eigen::Array<int, n_counters, 1>;
#pragma omp declare reduction(+ : Counters : omp_out = omp_out + omp_in) \
  initializer(omp_priv = Counters::Zero())

struct PartialResult
{
  int thread_id;
  Counters counters;

  PartialResult(Counters counters)
      : thread_id(omp_get_thread_num()), counters(counters)
  {
  }
};

struct LogMessage
{
  enum LogType
  {
    collisions,
    warnings,
    fragments,
    probabilities,
    intenergy,
    tmp,
    tmp_evolution,
    file_energy_distribution,
    final_position,
    pinhole
  };

  LogType type;
  std::string message;

  LogMessage(LogType type, std::string message)
      : type(type), message(message)
  {
  }

  LogMessage(LogType type, const char *message)
      : type(type), message(message)
  {
  }

  template <typename Callback>
  LogMessage(LogType type, Callback cb)
      : type(type), message(call_with_stringstream(cb))
  {
  }
};

struct LogFileWriter
{
  std::ofstream out_streams[10];

  LogFileWriter(char *file_probabilities)
  {
    if (LOGLEVEL >= LOGLEVEL_MIN)
    {
      this->open(LogMessage::collisions, Filenames::COLLISIONS);
      this->open(LogMessage::warnings, Filenames::WARNINGS, false);
      this->open(LogMessage::fragments, Filenames::FRAGMENTS);
      this->open(LogMessage::probabilities, file_probabilities);
      this->open(LogMessage::intenergy, Filenames::INTENERGY);
      this->open(LogMessage::tmp, Filenames::TMP);
      this->open(LogMessage::tmp_evolution, Filenames::TMP_EVOLUTION);
      this->open(LogMessage::file_energy_distribution, Filenames::ENERGY_DISTRIBUTION);
      this->open(LogMessage::final_position, Filenames::FINAL_POSITION);
      this->open(LogMessage::pinhole, Filenames::PINHOLE);
    }
  }

  void close()
  {
    for (auto &stream : this->out_streams)
    {
      if (stream.is_open())
      {
        stream.close();
      }
    }
  }

private:
  void
  open(LogMessage::LogType type, const char *const filename, bool scientific = true)
  {
    this->out_streams[type].open(filename);
    if (scientific)
    {
      this->out_streams[type] << std::setprecision(12) << std::scientific;
    }
  }
};

using StreamingResultElement = std::variant<std::monostate, LogMessage, PartialResult, std::exception>;
using StreamingResultQueue = BlockingConcurrentQueue<StreamingResultElement>;

struct WarningHelper
{
  Counters &counters;
  StreamingResultQueue &result_queue;

  template <typename T>
  void operator()(T msg)
  {
    counters[Counter::nwarnings] += 1;
    result_queue.enqueue(LogMessage(LogMessage::warnings, msg));
  }
};

struct LogHelper
{
  StreamingResultQueue &result_queue;
  LogMessage::LogType type;

  template <typename T>
  void operator()(T msg)
  {
    result_queue.enqueue(LogMessage(type, msg));
  }
};
