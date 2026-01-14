#pragma once

#include "messages.h"
#include "openmp_helper.h"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <variant>
#include <Eigen/Dense>
#include <magic_enum/magic_enum.hpp>
#include <blockingconcurrentqueue.h>

using magic_enum::enum_count;
using moodycamel::BlockingConcurrentQueue;

namespace Counter
{
enum Counter
{
  nwarnings,
  n_escaped_total,
  ncoll_total,
  counter_collision_rejections,
  n_fragmented_total
};
};
constexpr auto n_counters = enum_count<Counter::Counter>();

#pragma omp declare reduction(+ : Eigen::ArrayXi : omp_out = omp_out + omp_in) \
  initializer(omp_priv = Eigen::ArrayXi::Zero(omp_orig.size()))

struct PartialResult
{
  int thread_id;
  Eigen::ArrayXi counters;

  PartialResult(Eigen::ArrayXi counters)
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

using StreamingResultElement = std::variant<std::monostate, LogMessage, PartialResult, std::exception>;
using StreamingResultQueue = BlockingConcurrentQueue<StreamingResultElement>;

struct WarningHelper
{
  Eigen::ArrayXi &counters;
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
