#pragma once

#include "messages.h"
#include "operation_context.h"

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
  ncoll_total,
  counter_collision_rejections,
  n_realizations,
  n_escaped_total,
  n_fragmented_total
};
};
constexpr auto n_counters = enum_count<Counter::Counter>();

struct PartialResult
{
  Eigen::ArrayXi counters;

  PartialResult(Eigen::ArrayXi counters)
      : counters(std::move(counters))
  {
  }
};

struct ParticleStateMsg
{
  int realization;
  Eigen::Array4d postime;
  Eigen::Array3d velocity;
  Eigen::Array3d omega;
  double rot_energy;
  double internal_energy;
  int particle_index = 0;
};

struct CollisionEvent
{
  ParticleStateMsg state;
  double theta;
  double u_norm;
  bool accepted;
};

struct FragmentationEvent
{
  ParticleStateMsg state;
  int pathway_index;
  int next_particle_index = -1;
};

struct EscapeEvent
{
  ParticleStateMsg state;
};

using EventMessage = std::variant<CollisionEvent, FragmentationEvent, EscapeEvent>;

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
    pinhole,
    initial_trace,
  };

  LogType type;
  std::string message;

  template <typename Arg>
  LogMessage(LogType type, Arg arg)
      : type(type), message(prepare_message(arg))
  {
  }
};

using StreamingResultElement = std::variant<std::monostate, LogMessage, EventMessage, PartialResult>;
using StreamingResultQueue = BlockingConcurrentQueue<StreamingResultElement>;

struct WarningHelper
{
  Eigen::ArrayXi &counters;
  StreamingResultQueue &result_queue;
  OperationContext *operation = nullptr;

  template <typename T>
  void operator()(T msg)
  {
    if (operation != nullptr && !operation->should_continue())
    {
      return;
    }
    counters[Counter::nwarnings] += 1;
    result_queue.enqueue(LogMessage(LogMessage::warnings, msg));
  }
};

struct LogHelper
{
  StreamingResultQueue &result_queue;
  LogMessage::LogType type;
  OperationContext *operation = nullptr;

  template <typename T>
  void operator()(T msg)
  {
    if (operation != nullptr && !operation->should_continue())
    {
      return;
    }
    result_queue.enqueue(LogMessage(type, msg));
  }
};
