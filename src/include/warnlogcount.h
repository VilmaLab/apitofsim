#pragma once

#include "messages.h"
#include "particle_state.h"
#include "operation_context.h"

#include <fstream>
#include <functional>
#include <optional>
#include <iostream>
#include <iomanip>
#include <cstring>
#include <variant>
#include <atomic>
#include <climits>
#include <cstddef>
#include <limits>
#include <utility>
#include <oneapi/tbb/enumerable_thread_specific.h>
#include <Eigen/Dense>
#include <magic_enum/magic_enum.hpp>
#if __has_include(<moodycamel/blockingconcurrentqueue.h>)
#include <moodycamel/blockingconcurrentqueue.h>
#else
#include <blockingconcurrentqueue.h>
#endif

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
  ParticleState particle;
  double rot_energy;
  int particle_index = 0;

  Eigen::Array4d postime() const
  {
    return {particle.position.x(), particle.position.y(), particle.position.z(), particle.time};
  }
};

inline ParticleStateMsg snapshot(int realization, const ParticleState &particle, double inertia, int particle_index = 0)
{
  return {realization, particle, particle.rotational_energy(inertia), particle_index};
}

struct InitEvent
{
  ParticleStateMsg state;
};

struct CollisionEvent
{
  ParticleStateMsg state;
  double theta;
  double u_norm;
  bool accepted;
};

struct PostCollisionEvent
{
  ParticleStateMsg state;
};

struct FragmentationEvent
{
  ParticleStateMsg state;
  int pathway_index;
  int next_particle_index = -1;
};

struct PostFragmentationEvent
{
  ParticleStateMsg state;
};

struct EscapeEvent
{
  ParticleStateMsg state;
};

using EventMessage = std::variant<InitEvent, CollisionEvent, PostCollisionEvent, FragmentationEvent, PostFragmentationEvent, EscapeEvent>;

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
class StreamingResultQueue
{
  std::unique_ptr<BlockingConcurrentQueue<StreamingResultElement>> storage;
  struct ProducerCount
  {
    size_t count{0};
  };
  oneapi::tbb::enumerable_thread_specific<ProducerCount> producers;
  size_t consumed{0}; // Exactly one consuming thread per simulation.
  size_t capacity;
  size_t chunk;
  std::atomic<std::ptrdiff_t> pending{0};
  std::atomic<bool> bounded;

public:
  using Sink = std::function<void(StreamingResultElement &&)>;
  // The simulation body runs synchronously on a worker for one realization.
  inline static thread_local std::vector<StreamingResultElement> *token_items = nullptr;
  std::optional<bool> pipeline_order;
  bool pipeline_logs = false;
  Sink pipeline_sink;

  explicit StreamingResultQueue(size_t capacity = 0, bool queued = true) : capacity(capacity), chunk(std::min(size_t{1024}, std::max(size_t{1}, capacity / 2))), bounded(capacity != 0)
  {
    if (capacity > static_cast<size_t>(INT_MAX / 2))
      throw std::invalid_argument("Event queue capacity is too large");
    if (queued)
      storage = std::make_unique<BlockingConcurrentQueue<StreamingResultElement>>();
  }

  template <typename T>
  bool enqueue(T &&item)
  {
    if (pipeline_order.has_value())
    {
      if constexpr (std::is_same_v<std::remove_cvref_t<T>, LogMessage>)
      {
        if (!pipeline_logs)
          return true;
      }
      if (pipeline_sink)
      {
        if (token_items != nullptr)
          token_items->emplace_back(std::forward<T>(item));
        else
          pipeline_sink(StreamingResultElement(std::forward<T>(item)));
      }
      return true;
    }
    return enqueue_for_delivery(std::forward<T>(item));
  }

  template <typename T>
  bool enqueue_for_delivery(T &&item)
  {
    if (!storage->enqueue(std::forward<T>(item)))
      throw std::bad_alloc();
    if (bounded.load(std::memory_order_relaxed))
    {
      auto &producer = producers.local();
      if (++producer.count == chunk)
      {
        producer.count = 0;
        auto count = pending.fetch_add(static_cast<std::ptrdiff_t>(chunk)) + static_cast<std::ptrdiff_t>(chunk);
        while (std::cmp_greater(count, capacity) && bounded.load())
        {
          pending.wait(count);
          count = pending.load();
        }
      }
    }
    return true;
  }

  void release_slot()
  {
    // Credits are amortized over blocks. The bound allows one partially
    // accounted block per producer, avoiding an atomic RMW on every event.
    if (bounded.load(std::memory_order_relaxed) && ++consumed == chunk)
    {
      consumed = 0;
      pending.fetch_sub(static_cast<std::ptrdiff_t>(chunk));
      pending.notify_all();
    }
  }

  bool try_dequeue(StreamingResultElement &item)
  {
    if (!storage->try_dequeue(item))
      return false;
    release_slot();
    return true;
  }

  void wait_dequeue(StreamingResultElement &item)
  {
    storage->wait_dequeue(item);
    release_slot();
  }

  void unblock_producers()
  {
    bounded = false;
    // Atomic wait requires a changed value, not just a notification. A distant
    // negative sentinel also prevents an ABA from in-flight producer updates.
    pending = std::numeric_limits<std::ptrdiff_t>::min() / 2;
    pending.notify_all();
  }
};

struct DrainingStreamingResultQueue
{
  StreamingResultQueue queue;
  bool draining = false;

  explicit DrainingStreamingResultQueue(size_t capacity = 0, bool queued = true) : queue(capacity, queued)
  {
  }

  void draining_dequeue(StreamingResultElement &result)
  {
    if (!draining)
    {
      queue.wait_dequeue(result);
      if (std::holds_alternative<std::monostate>(result))
      {
        draining = true;
      }
      else
      {
        return;
      }
    }

    bool got = queue.try_dequeue(result);
    if (!got)
    {
      result = std::monostate{};
    }
  }
};

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
    if (!result_queue.pipeline_order.has_value() || result_queue.pipeline_logs)
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
    if (!result_queue.pipeline_order.has_value() || result_queue.pipeline_logs)
      result_queue.enqueue(LogMessage(type, msg));
  }
};
