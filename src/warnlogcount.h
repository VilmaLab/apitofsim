#pragma once

#include "utils.h"
#include "omp_shim.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <variant>
#include <Eigen/Dense>
#include <magic_enum/magic_enum.hpp>
#pragma clang attribute push(__attribute__((no_sanitize("unsigned-integer-overflow"))), apply_to = function)
#include <blockingconcurrentqueue.h>
#pragma clang attribute pop

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

std::string prepare_message(const std::string &msg)
{
  return msg;
}

std::string prepare_message(const char *msg)
{
  return msg;
}

template <typename Callback>
std::string prepare_message(Callback cb)
{
  return call_with_stringstream(cb);
}

class ApiTofError : public std::runtime_error
{
public:
  template <typename Arg>
  ApiTofError(Arg arg)
      : std::runtime_error(prepare_message(arg))
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

  template <typename Arg>
  LogMessage(LogType type, Arg arg)
      : type(type), message(prepare_message(arg))
  {
  }
};

struct LogFileWriter
{
  std::ofstream out_streams[10];

  LogFileWriter(char *file_probabilities)
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
