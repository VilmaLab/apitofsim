#include <Python.h>

#include <cassert>
#include <atomic>
#include <cstdio>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <cstdlib>
#include <random>
#include <condition_variable>
#include <chrono>
#include <deque>

#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/chrono.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <oneapi/tbb/global_control.h>

#include "skimmer.h"
#include "densityandrate.h"
#include "mass_spec.h"
#include "trajectory.h"
#include "consts.h"
#include "warnlogcount.h"
#include "operation_context.h"
#include "tree_bindings.h"
#include "event_batch.h"
#include "event_recording.h"

using namespace std;

namespace nb = nanobind;
using namespace nb::literals;

namespace
{
// Py_AtExit owns cleanup; a static destructor could run after TBB teardown.
oneapi::tbb::task_scheduler_handle *tbb_scheduler_handle = nullptr;

void finalize_tbb() noexcept
{
  if (tbb_scheduler_handle == nullptr)
  {
    return;
  }
  const bool finalized = oneapi::tbb::finalize(*tbb_scheduler_handle, std::nothrow);
  delete tbb_scheduler_handle;
  tbb_scheduler_handle = nullptr;
  if (!finalized)
  {
    // Python has already finalized, so diagnostics must use native APIs only.
    std::fputs("apitofsim: could not finalize oneTBB; other scheduler users remain active.\n", stderr);
  }
}
} // namespace

using SkimmerResult = Eigen::Array<double, Eigen::Dynamic, 6>;

const unsigned long long DEFAULT_SEED = 42ULL;
const std::tuple<int, bool> DEFAULT_LOGCONF_TUPLE = std::tuple(DEFAULT_LOGLEVEL, false);

struct PythonWarningHelper
{
  template <typename Arg>
  void operator()(Arg msg)
  {
    PyErr_WarnEx(PyExc_Warning, prepare_message(msg).c_str(), 1);
  }
};

SkimmerResult skimmer(
  double T0,
  double P0,
  double rmax,
  double dc,
  double alpha_factor,
  Gas gas,
  int N,
  int M,
  int resolution,
  double tolerance)
{
  int nwarnings = 0;
  std::ofstream warnings;
  warnings.open("warnings_skimmer.dat");
  warnings << std::scientific;
  Skimmer s = {
    T0,
    P0,
    rmax,
    dc,
    alpha_factor,
    gas.mass,
    gas.adiabatic_index,
    N,
    M,
    resolution,
    tolerance,
    nwarnings,
    warnings,
  };

  SkimmerResult result(resolution, 6);

  int i = 0;
  while (true)
  {
    s.next();
    auto r = s.get();
    if (r.has_value())
    {
      result(i, 0) = r->r;
      result(i, 1) = r->vel;
      result(i, 2) = r->T;
      result(i, 3) = r->P;
      result(i, 4) = r->rho;
      result(i, 5) = r->speed_of_sound;
      i++;
    }
    else
    {
      break;
    }
  }
  return result;
}

std::tuple<Histogram, Histogram> densityandrate(
  const MolecularCluster &cluster_0,
  const MolecularCluster &cluster_1,
  const ClusterSpec &cluster_2,
  double energy_max,
  double energy_max_rate,
  double bin_width,
  double fragmentation_energy,
  bool k_rate_cancellation)
{
  // TODO:: The individual product densities are not used -- should be possible to not compute them
  const Products products = make_products(cluster_1, cluster_2);
  const ClusterDerived derived_1 = compute_derived(cluster_1);
  const ClusterDerived derived_2 = compute_derived(cluster_2);
  DensityResult rhos = compute_density_of_states_all(cluster_0, products, energy_max, bin_width);
  const Eigen::ArrayXd k_rate = compute_k_total_full(
    cluster_0,
    products,
    derived_1,
    derived_2,
    rhos,
    fragmentation_energy,
    energy_max_rate,
    bin_width,
    k_rate_cancellation);
  int m_max_rate = static_cast<int>(energy_max_rate / bin_width);
  int m_max = static_cast<int>(energy_max / bin_width);
  auto energies = prepare_energies(bin_width, m_max);
  auto energies_rate = prepare_energies(bin_width, m_max_rate);
  return {Histogram(energies, rhos.col(COMB_ROW)), Histogram(energies_rate, k_rate)};
}

unsigned long long root_seed(unsigned long long seed)
{
  mt19937 root_gen = mt19937(seed);
  return root_gen();
}

Eigen::ArrayXi mk_partial_counters(const MassSpecSubstanceSingleInput &subs)
{
  int total_counters = n_counters - 1 + subs.pathways.size();
  return Eigen::ArrayXi::Zero(total_counters);
}

Eigen::ArrayXi mk_partial_counters(const MassSpecSubstanceTreeInput &subs)
{
  int total_counters = n_counters - 1 + subs.tree_pathways.size();
  return Eigen::ArrayXi::Zero(total_counters);
}

/* Caller must ensure that all parameters passed as reference outlive thread */
template <typename MassSpecSubstanceT>
std::jthread run_mass_spec_in_thread(
  SimulationResult &result,
  OperationContext &operation,
  ExceptionTransport &exception_transport,
  const MassSpectrometer &ms,
  const MassSpecSubstanceT &subs,
  int N,
  unsigned long long seed,
  StreamingResultQueue &result_queue,
  SampleMode sample_mode,
  bool strict,
  std::tuple<int, bool> logconf,
  CollisionTimeOptions collision_time,
  std::function<void()> on_complete = {})
{
  return std::jthread([&, N, seed, sample_mode, strict, logconf, collision_time, on_complete]
  {
    exception_transport.guard([&, N, seed, sample_mode, strict, logconf, collision_time]
    {
      result = apitof_mass_spec(
        ms,
        subs,
        N,
        root_seed(seed),
        result_queue,
        sample_mode,
        strict,
        MassSpecLogConf(std::get<0>(logconf), std::get<1>(logconf)),
        operation, collision_time);
    });
    if (on_complete)
    {
      on_complete();
    }
    result_queue.enqueue(std::monostate{});
  });
}

using PumpedMassSpecItem = std::variant<std::tuple<const std::string, const std::string>, Eigen::ArrayXi, EventMessage, std::monostate>;
using MassSpecItem = std::variant<std::tuple<const std::string, const std::string>, Eigen::ArrayXi, SimulationResult>;

PumpedMassSpecItem convert_mass_spec_item(const StreamingResultElement &result, Eigen::ArrayXi &partial_counters)
{
  using magic_enum::enum_name;
  if (std::holds_alternative<std::monostate>(result))
  {
    return std::monostate{};
  }
  else if (std::holds_alternative<PartialResult>(result))
  {
    const PartialResult &partial_result = std::get<PartialResult>(result);
    partial_counters += partial_result.counters;
    return partial_counters;
  }
  else if (std::holds_alternative<LogMessage>(result))
  {
    const LogMessage &msg = std::get<LogMessage>(result);
    const std::string name = std::string(enum_name(msg.type));
    return std::tuple<const std::string, const std::string>(name, msg.message);
  }
  else if (std::holds_alternative<EventMessage>(result))
  {
    return std::get<EventMessage>(result);
  }
  else
  {
    throw ApiTofError("Unknown variant from mass spec output queue");
  }
}

PumpedMassSpecItem pump_mass_spec_queue(DrainingStreamingResultQueue &result_queue, Eigen::ArrayXi &partial_counters)
{
  StreamingResultElement result;
  result_queue.draining_dequeue(result);
  return convert_mass_spec_item(result, partial_counters);
}

template <typename MassSpecSubstanceT>
SimulationResult
mass_spec(
  const MassSpectrometer &ms,
  const MassSpecSubstanceT &subs,
  int N,
  unsigned long long seed = DEFAULT_SEED,
  std::optional<std::function<void(std::string_view, std::string)>> log_callback = nullopt,
  std::optional<std::function<void(Eigen::ArrayXi)>> result_callback = nullopt,
  SampleMode sample_mode = SampleMode::rejection,
  bool strict = true,
  std::tuple<int, bool> logconf = DEFAULT_LOGCONF_TUPLE,
  std::optional<CollisionTimeOptions> collision_time = std::nullopt,
  bool ordered = false)
{
  DrainingStreamingResultQueue result_queue(0, log_callback.has_value() || result_callback.has_value());
  OperationContext operation;
  result_queue.queue.pipeline_order = ordered;
  result_queue.queue.pipeline_logs = log_callback.has_value();
  // This API returns counters and optionally calls Python on the caller's
  // thread. It has no event delivery sink.
  std::get<1>(logconf) = false;
  if (!log_callback && !result_callback)
    return operation.run([&]
    {
      return apitof_mass_spec(ms, subs, N, root_seed(seed), result_queue.queue,
                              sample_mode, strict, MassSpecLogConf(std::get<0>(logconf), false),
                              operation, collision_time.value_or(CollisionTimeOptions{}));
    });
  result_queue.queue.pipeline_sink = [&](StreamingResultElement &&item)
  {
    if (result_callback || !std::holds_alternative<PartialResult>(item))
      result_queue.queue.enqueue_for_delivery(std::move(item));
  };
  ExceptionTransport exception_transport;
  SimulationResult result;
  Eigen::ArrayXi partial_counters = mk_partial_counters(subs);
  std::jthread thread = run_mass_spec_in_thread<MassSpecSubstanceT>(result, operation, exception_transport, ms, subs, N, seed, result_queue.queue, sample_mode, strict, logconf, collision_time.value_or(CollisionTimeOptions{}));
  while (true)
  {
    auto result = pump_mass_spec_queue(result_queue, partial_counters);
    if (std::holds_alternative<std::tuple<const std::string, const std::string>>(result))
    {
      auto tpl = std::get<std::tuple<const std::string, const std::string>>(result);
      if (log_callback)
      {
        (*log_callback)(std::get<0>(tpl), std::get<1>(tpl));
      }
    }
    else if (std::holds_alternative<Eigen::ArrayXi>(result))
    {
      if (result_callback)
      {
        (*result_callback)(std::get<Eigen::ArrayXi>(result));
      }
    }
    else if (std::holds_alternative<EventMessage>(result))
    {
      // Event inspection uses the columnar iterator.
    }
    else if (std::holds_alternative<std::monostate>(result))
    {
      break;
    }
    else
    {
      throw ApiTofError("Unknown variant from mass spec output queue");
    }
  }
  operation.rethrow_pending_signal();
  exception_transport.rethrow();
  return result;
}

struct MassSpecBatch
{
  std::shared_ptr<event_batch::Columns> events;
  std::vector<MassSpecItem> items;
};

struct StreamDelivery
{
  std::shared_ptr<IpcEventWriter> writer;
  int64_t result_id;
  bool events;
  bool ordered;
  bool progress;
  bool logs;
};

struct MassSpecIterator
{
  DrainingStreamingResultQueue result_queue;
  Eigen::ArrayXi partial_counters;
  OperationContext operation;
  ExceptionTransport exception_transport;
  SimulationResult final_result;
  bool finished{false};
  std::atomic<bool> completed{false};
  std::atomic<bool> cancelled{false};
  int requested_realizations;
  std::mutex join_mutex;
  std::array<std::shared_ptr<event_batch::Columns>, 2> buffers;
  std::vector<StreamingResultElement> raw_buffer;
  size_t initial_batch_size;
  size_t max_batch_size;
  bool first_batch{true};
  std::shared_ptr<const MassSpectrometer> raw_ms;
  std::shared_ptr<const void> raw_subs;
  std::mutex delivery_mutex;
  std::condition_variable delivery_changed;
  std::deque<MassSpecBatch> delivery;
  std::exception_ptr writer_error;
  bool pump_done{false};
  const bool has_pump;
  bool discard_delivery{false};
  const bool deliver_events;
  size_t delivery_capacity;
  size_t delivery_rows = 0;
  std::optional<std::array<int64_t, 3>> recording_range;
  std::jthread execution_thread;

  static size_t validate_batch_size(size_t initial, size_t maximum)
  {
    if (initial == 0 || maximum == 0 || initial > maximum)
      throw std::invalid_argument("Batch sizes must be positive and initial must not exceed maximum");
    return initial;
  }

  template <typename MassSpecSubstanceT>
  MassSpecIterator(
    std::shared_ptr<const MassSpectrometer> ms,
    std::shared_ptr<const MassSpecSubstanceT> subs,
    int N,
    unsigned long long seed = DEFAULT_SEED,
    SampleMode sample_mode = SampleMode::rejection,
    bool strict = true,
    std::tuple<int, bool> logconf = DEFAULT_LOGCONF_TUPLE,
    std::optional<CollisionTimeOptions> collision_time = std::nullopt,
    size_t initial_batch_size = 4096,
    size_t max_batch_size = 262144,
    size_t queue_capacity = 0,
    std::optional<StreamDelivery> stream = std::nullopt) : result_queue(queue_capacity, !stream),
                                                           partial_counters(mk_partial_counters(*subs)),
                                                           requested_realizations(N),
                                                           initial_batch_size(validate_batch_size(initial_batch_size, max_batch_size)),
                                                           max_batch_size(max_batch_size),
                                                           raw_ms(ms),
                                                           raw_subs(subs),
                                                           has_pump(stream.has_value()),
                                                           deliver_events(stream && stream->events),
                                                           delivery_capacity(queue_capacity)
  {
    if (stream)
    {
      if (stream->result_id < 0)
        throw std::invalid_argument("A stream requires a nonnegative result ID");
      std::optional<IpcEventWriter::Lease> writer;
      if (stream->writer)
        writer.emplace(stream->writer->acquire());
      auto execute = [this, instrument = ms.get(), substance = subs.get(), N, seed, sample_mode, strict, logconf, collision_time]
      {
        return apitof_mass_spec(*instrument, *substance, N, root_seed(seed), result_queue.queue,
                                sample_mode, strict, MassSpecLogConf(std::get<0>(logconf), std::get<1>(logconf)),
                                operation, collision_time.value_or(CollisionTimeOptions{}));
      };
      execution_thread = std::jthread(
        [this, execute, result_id = stream->result_id, ordered = stream->ordered,
         progress = stream->progress, logs = stream->logs, writer = std::move(writer)]() mutable
      { pump(execute, result_id, ordered, progress, logs, writer); });
    }
    else
    {
      // Keep the inputs alive for the raw iterator, which can start before a
      // consumer is attached and supports joining before reading.
      execution_thread = run_mass_spec_in_thread<MassSpecSubstanceT>(final_result, operation, exception_transport, *ms, *subs, N, seed, result_queue.queue, sample_mode, strict, logconf, collision_time.value_or(CollisionTimeOptions{}), [this]
      { completed = true; });
    }
  }

  MassSpecBatch __next__()
  {
    return read_batch(max_batch_size);
  }

  MassSpecItem decode_result(PumpedMassSpecItem &result)
  {
    if (std::holds_alternative<std::monostate>(result))
    {
      finished = true;
      {
        nb::gil_scoped_release release;
        join_if_joinable();
      }
      operation.rethrow_pending_signal();
      exception_transport.rethrow();
      // A cancelled TBB reduction may omit completed branches. The drained
      // partial results contain every completed realization exactly once.
      std::get<0>(final_result) = partial_counters;
      cancelled = !operation.checkpoint() && partial_counters[Counter::n_realizations] < requested_realizations;
      return final_result;
    }
    else if (std::holds_alternative<Eigen::ArrayXi>(result))
    {
      return std::get<Eigen::ArrayXi>(result);
    }
    else if (std::holds_alternative<std::tuple<const std::string, const std::string>>(result))
    {
      return std::get<std::tuple<const std::string, const std::string>>(result);
    }
    else
    {
      throw ApiTofError("Unknown variant from mass spec output queue");
    }
  }

  MassSpecBatch read_batch(size_t max_items)
  {
    if (has_pump)
      return read_recorded_batch();
    if (max_items == 0)
    {
      throw std::invalid_argument("max_items must be positive");
    }
    if (finished)
    {
      throw nb::stop_iteration();
    }
    max_items = std::min(max_items, first_batch ? initial_batch_size : max_batch_size);
    first_batch = false;
    auto &raw = raw_buffer;
    raw.clear();
    raw.reserve(max_items);
    {
      nb::gil_scoped_release release;
      StreamingResultElement first;
      result_queue.draining_dequeue(first);
      raw.push_back(std::move(first));
      if (!std::holds_alternative<std::monostate>(raw.front()))
      {
        while (raw.size() < max_items)
        {
          StreamingResultElement item;
          if (!result_queue.queue.try_dequeue(item))
          {
            break;
          }
          if (std::holds_alternative<std::monostate>(item))
          {
            // Terminal marker may precede other producer queues. Drain them on
            // later reads too; do not raise a terminal error after popping data.
            result_queue.draining = true;
            continue;
          }
          raw.push_back(std::move(item));
        }
      }
    }
    std::shared_ptr<event_batch::Columns> columns;
    for (auto &buffer : buffers)
    {
      if (!buffer)
        buffer = std::make_shared<event_batch::Columns>();
      if (buffer.use_count() == 1)
      {
        columns = buffer;
        break;
      }
    }
    // User-retained batches may occupy both reusable buffers. Their ownership
    // takes precedence over reuse; only two idle buffers are retained here.
    if (!columns)
      columns = std::make_shared<event_batch::Columns>();
    MassSpecBatch output{columns, {}};
    {
      nb::gil_scoped_release release;
      columns->clear(raw.size());
      for (const auto &item : raw)
      {
        if (const auto *event = std::get_if<EventMessage>(&item))
          columns->append(*event);
      }
    }
    for (auto &item : raw)
    {
      if (!std::holds_alternative<EventMessage>(item))
      {
        auto converted = convert_mass_spec_item(item, partial_counters);
        output.items.push_back(decode_result(converted));
      }
    }
    return output;
  }

  void join_if_joinable()
  {
    if (has_pump)
      require_stopped();
    join_worker();
  }

  void join_worker()
  {
    std::scoped_lock lock(join_mutex);
    if (execution_thread.joinable())
    {
      execution_thread.join();
    }
  }

  void request_cancel()
  {
    if (!completed)
    {
      operation.request_cancel();
    }
    result_queue.queue.unblock_producers();
    delivery_changed.notify_all();
  }

  void close()
  {
    request_cancel();
    if (has_pump)
    {
      {
        std::scoped_lock lock(delivery_mutex);
        discard_delivery = true;
        delivery.clear();
        delivery_rows = 0;
      }
      delivery_changed.notify_all();
    }
    join_worker();
  }

  void pump(const auto &execute, int64_t result_id, bool ordered, bool progress, bool logs,
            std::optional<IpcEventWriter::Lease> &writer)
  {
    const auto start = writer ? writer->counts() : std::array<int64_t, 2>{};
    try
    {
      const bool capture_events = writer || deliver_events;
      std::vector<int64_t> ordinals(capture_events ? static_cast<size_t>(std::max(0, requested_realizations)) : 0, 0);
      auto columns = std::make_shared<EventColumns>();
      const auto maximum = std::min(max_batch_size, size_t{65536});
      auto limit = std::min(initial_batch_size, maximum);
      if (capture_events)
        columns->clear(limit);
      std::vector<MassSpecItem> items;
      bool updated = false;

      auto flush = [&]
      {
        const auto rows = columns->event_indices.size();
        bool handed_to_writer = false;
        if (writer && rows != 0)
        {
          try
          {
            handed_to_writer = writer->write(columns, result_id);
          }
          catch (...)
          {
            writer_error = std::current_exception();
            throw;
          }
        }
        if (updated && progress)
          items.emplace_back(partial_counters);
        updated = false;
        const bool has_delivery = (deliver_events && rows != 0) || !items.empty();
        if (has_delivery)
        {
          auto delivered = deliver_events ? columns : std::make_shared<EventColumns>();
          if (deliver_events && delivered->result_ids.empty())
            delivered->result_ids.assign(rows, result_id);
          std::unique_lock lock(delivery_mutex);
          // A single batch may exceed the credit limit. Let it through
          // when the handoff is empty so recording batch sizes stay stable.
          const auto pending_rows = deliver_events ? rows : 0;
          while (!discard_delivery && operation.checkpoint() &&
                 (delivery.size() >= 2 ||
                  (delivery_rows != 0 && delivery_capacity != 0 && delivery_rows + pending_rows > delivery_capacity)))
            delivery_changed.wait_for(lock, std::chrono::milliseconds(50));
          if (!discard_delivery)
          {
            delivery.push_back(MassSpecBatch{std::move(delivered), std::move(items)});
            delivery_rows += pending_rows;
          }
          lock.unlock();
          delivery_changed.notify_all();
          items.clear();
        }
        if (handed_to_writer || (deliver_events && has_delivery))
          columns = std::make_shared<EventColumns>();
        if (rows != 0)
          limit = maximum;
        if (capture_events)
          columns->clear(limit);
      };

      auto consume = [&](const StreamingResultElement &item)
      {
        if (const auto *event = std::get_if<EventMessage>(&item))
        {
          if (!capture_events)
            return;
          columns->append(*event);
          const auto realization = columns->indices[0].back();
          if (realization < 0 || static_cast<size_t>(realization) >= ordinals.size())
            throw std::runtime_error("Native event has an invalid realization index");
          columns->event_indices.push_back(ordinals[realization]++);
          if (columns->event_indices.size() == limit)
            flush();
        }
        else if (const auto *partial = std::get_if<PartialResult>(&item))
        {
          partial_counters += partial->counters;
          updated = true;
          if (!capture_events && progress)
            flush();
        }
        else if (logs && std::holds_alternative<LogMessage>(item))
        {
          auto value = convert_mass_spec_item(item, partial_counters);
          items.emplace_back(std::get<std::tuple<const std::string, const std::string>>(value));
          if (!capture_events)
            flush();
        }
      };

      result_queue.queue.pipeline_order = ordered || writer.has_value();
      result_queue.queue.pipeline_logs = logs;
      if (capture_events || progress || logs)
        result_queue.queue.pipeline_sink = consume;
      exception_transport.guard([&]
      { final_result = execute(); });
      result_queue.queue.pipeline_sink = {};
      if (!progress && !capture_events && !logs && exception_transport.should_continue())
        partial_counters = std::get<0>(final_result);
      if (!writer_error)
        flush();
      std::get<0>(final_result) = partial_counters;
      cancelled = !operation.checkpoint() && partial_counters[Counter::n_realizations] < requested_realizations;
    }
    catch (...)
    {
      writer_error = std::current_exception();
      request_cancel();
    }
    if (writer)
    {
      const auto end = writer->counts();
      recording_range = {start[0], end[0] - start[0], end[1] - start[1]};
      writer.reset();
    }
    completed = true;
    {
      std::scoped_lock lock(delivery_mutex);
      pump_done = true;
    }
    delivery_changed.notify_all();
  }

  MassSpecBatch read_recorded_batch()
  {
    if (finished)
      throw nb::stop_iteration();
    MassSpecBatch batch{std::make_shared<EventColumns>(), {}};
    {
      nb::gil_scoped_release release;
      std::unique_lock lock(delivery_mutex);
      delivery_changed.wait(lock, [this]
      { return !delivery.empty() || pump_done; });
      if (!delivery.empty())
      {
        batch = std::move(delivery.front());
        delivery.pop_front();
        delivery_rows -= batch.events->event_indices.size();
        delivery_changed.notify_all();
        return batch;
      }
      lock.unlock();
      join_worker();
    }
    finished = true;
    batch.items.emplace_back(run_result());
    return batch;
  }

  void require_stopped()
  {
    std::scoped_lock lock(delivery_mutex);
    if (!has_pump || !pump_done)
      throw std::logic_error("Drain or close the stream before reading its outcome");
  }

  SimulationResult run_result()
  {
    require_stopped();
    if (writer_error)
      std::rethrow_exception(writer_error);
    operation.rethrow_pending_signal();
    exception_transport.rethrow();
    return final_result;
  }

  ~MassSpecIterator()
  {
    close();
  }
};

template <typename SamplerT, typename GenT>
Eigen::ArrayX2d dispatch_sample_collision(
  SamplerT sampler,
  GenT gen,
  int num_samples,
  double v_rel_norm,
  double mobility_gas,
  double mobility_gas_inv,
  double R_tot,
  double n)
{
  auto warn = PythonWarningHelper();
  Eigen::ArrayX2d samples(num_samples, 2);
  for (int i = 0; i < num_samples; i++)
  {
    std::tie(samples(i, 0), samples(i, 1)) = sampler.sample(gen, n, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, warn);
  }
  return samples;
}

Eigen::ArrayX2d sample_collision(
  SampleMode sample_mode,
  int num_samples,
  double v_rel_norm,
  Gas gas,
  double R_cluster,
  double P,
  double T,
  unsigned long long seed,
  double dtheta,
  std::optional<double> du)
{
  mt19937 gen = mt19937(seed);
  double kT = consts::boltzmann * T;
  double mobility_gas = kT / gas.mass;
  double mobility_gas_inv = gas.mass / kT;
  double boundary_u = 5.0 * sqrt(mobility_gas);
  double R_tot = gas.radius + R_cluster;
  double n = particle_density(P, T);
  double du_val;
  if (du.has_value())
  {
    du_val = *du;
  }
  else
  {
    du_val = 1.0e-4 * sqrt(mobility_gas);
  }
  if (sample_mode == SampleMode::dss_normalized)
  {
    auto sampler = GasCollCondNormHistDSSSampler(dtheta, du_val, boundary_u);
    return dispatch_sample_collision(sampler, gen, num_samples, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, n);
  }
  else if (sample_mode == SampleMode::dss_unnormalized)
  {
    auto sampler = GasCollCondUnnormHistDSSSampler(dtheta, du_val, boundary_u);
    return dispatch_sample_collision(sampler, gen, num_samples, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, n);
  }
  else if (sample_mode == SampleMode::rejection)
  {
    auto sampler = GasCollRejectionSampler(boundary_u);
    return dispatch_sample_collision(sampler, gen, num_samples, v_rel_norm, mobility_gas, mobility_gas_inv, R_tot, n);
  }
  else
  {
    throw ApiTofArgumentError([&](auto &msg)
    {
      msg << "Unknown sampling mode: " << static_cast<int>(sample_mode) << std::endl;
    });
  }
}

template <typename EnumT>
void nb_magic_enum(nanobind::handle scope, const char *name)
{
  auto enum_wrap = nb::enum_<EnumT>(scope, name);
  for (auto entry : magic_enum::enum_entries<EnumT>())
  {
    enum_wrap.value(entry.second.data(), entry.first);
  }
  enum_wrap.export_values();
}

template <typename CppExceptionT>
void register_overflow_translator(nb::exception<CppExceptionT> nb_py_exception)
{
  nb::register_exception_translator(
    [](const std::exception_ptr &exc, void *payload)
  {
    try
    {
      std::rethrow_exception(exc);
    }
    catch (const CppExceptionT &err)
    {
      auto *c_py_exc = static_cast<PyObject *>(payload);
      auto py_exc = nb::borrow(c_py_exc)(err.what());
      py_exc.attr("max") = err.max;
      py_exc.attr("current") = err.current;
      PyErr_SetObject(c_py_exc, py_exc.ptr());
    }
  }, nb_py_exception.ptr());
}

NB_MODULE(apitofsimraw, m)
{
  if (tbb_scheduler_handle == nullptr)
  {
    auto handle = std::make_unique<oneapi::tbb::task_scheduler_handle>(oneapi::tbb::attach{});
    if (Py_AtExit(finalize_tbb) != 0)
    {
      throw std::runtime_error("Could not register oneTBB interpreter-exit finalization");
    }
    tbb_scheduler_handle = handle.release();
  }

  m.doc() = "APi-TOF-MS simulation module";
  m.def("skimmer", &skimmer);

  nb::class_<AtomLikeCluster>(m, "AtomLikeCluster")
    .def(nb::init<int, double, int>(),
         "nucleon_count"_a,
         "electronic_energy"_a,
         "charge"_a)
    .def_ro("nucleon_count", &AtomLikeCluster::nucleon_count)
    .def_ro("electronic_energy", &AtomLikeCluster::electronic_energy)
    .def_ro("charge", &AtomLikeCluster::charge);

  nb::class_<MolecularCluster>(m, "MolecularCluster")
    .def(nb::init<int, double, Eigen::Vector3d, Eigen::ArrayXd, int>(),
         "nucleon_count"_a,
         "electronic_energy"_a,
         "rotations"_a,
         "frequencies"_a,
         "charge"_a)
    .def_ro("nucleon_count", &MolecularCluster::nucleon_count)
    .def_ro("electronic_energy", &MolecularCluster::electronic_energy)
    .def_ro("rotations", &MolecularCluster::rotations)
    .def_ro("frequencies", &MolecularCluster::frequencies)
    .def_ro("charge", &MolecularCluster::charge)
    .def("num_oscillators", &MolecularCluster::num_oscillators);

  nb::class_<ClusterDerived>(m, "ClusterDerived")
    .def_ro("inertia_moment", &ClusterDerived::inertia_moment)
    .def_ro("radius", &ClusterDerived::radius)
    .def_ro("mass", &ClusterDerived::mass);

  m.def("num_oscillators", &num_oscillators, "spec"_a);
  m.def("compute_derived", &compute_derived, "spec"_a);

  nb::class_<Gas>(m, "Gas")
    .def(nb::init<double, double, double>(),
         nb::arg("radius"),
         nb::arg("mass"),
         nb::arg("adiabatic_index"))
    .def_ro("radius", &Gas::radius)
    .def_ro("mass", &Gas::mass)
    .def_ro("adiabatic_index", &Gas::adiabatic_index);

  nb::class_<Histogram>(m, "Histogram")
    .def(nb::init<Eigen::ArrayXd, Eigen::ArrayXd>(),
         nb::arg("x"),
         nb::arg("y"))
    .def(nb::init<double, int, Eigen::ArrayXd>(),
         nb::arg("bin_width"),
         nb::arg("m_max"),
         nb::arg("y"))
    .def_prop_ro("x", &Histogram::x)
    .def_prop_ro("y", &Histogram::y)
    .def_prop_ro("x_max", &Histogram::x_max)
    .def_prop_ro("bin_width", &Histogram::bin_width);

  nb::class_<Quadrupole>(m, "Quadrupole")
    .def(nb::init<double, double, double, double>(),
         nb::arg("dc_field"),
         nb::arg("ac_field"),
         nb::arg("radiofrequency"),
         nb::arg("r_quadrupole"))
    .def_ro("dc_field", &Quadrupole::dc_field)
    .def_ro("ac_field", &Quadrupole::ac_field)
    .def_ro("radiofrequency", &Quadrupole::radiofrequency)
    .def_ro("r_quadrupole", &Quadrupole::r_quadrupole);

  nb::class_<MassSpectrometer>(m, "MassSpectrometer")
    .def(nb::init<SkimmerData, double, InstrumentDims, InstrumentVoltages, double, InstrumentPressures, std::optional<Quadrupole>, std::optional<double>>(),
         "skimmer"_a,
         "mesh_skimmer"_a,
         "lengths"_a,
         "voltages"_a,
         "T"_a,
         "pressures"_a,
         "quadrupole"_a = std::nullopt,
         "radius_pinhole"_a.none() = 0.001)
    .def_ro("skimmer", &MassSpectrometer::skimmer)
    .def_ro("mesh_skimmer", &MassSpectrometer::mesh_skimmer)
    .def_ro("lengths", &MassSpectrometer::lengths)
    .def_ro("voltages", &MassSpectrometer::voltages)
    .def_ro("T", &MassSpectrometer::T)
    .def_ro("pressures", &MassSpectrometer::pressures)
    .def_ro("quadrupole", &MassSpectrometer::quadrupole)
    .def_ro("radius_pinhole", &MassSpectrometer::radius_pinhole);

  nb::class_<CollisionProfile>(m, "CollisionProfile")
    .def(nb::init<const MassSpectrometer &, const Gas &, double>(), "mass_spec"_a, "gas"_a, "cluster_radius"_a)
    .def("frequency", &CollisionProfile::frequency, "z"_a, "velocity"_a);

  nb_magic_enum<FragmentationStateMethod>(m, "FragmentationStateMethod");
  nb::class_<FragmentationDos>(m, "FragmentationDos")
    .def(nb::init<const Eigen::ArrayXd &, double, double, std::optional<Eigen::ArrayXd>>(),
         "frequencies"_a, "bin_width"_a, "energy_max"_a, "log_counts"_a = nb::none())
    .def_prop_ro("log_counts", [](const FragmentationDos &dos)
  { return Eigen::ArrayXd(dos.counts()); })
    .def("extend", &FragmentationDos::extend, "frequencies"_a);
  nb::class_<FragmentationProducts>(m, "FragmentationProducts")
    .def(nb::init<const MolecularCluster &, const ClusterSpec &, double, double, std::optional<Eigen::ArrayXXd>>(),
         "daughter"_a, "neutral"_a, "bin_width"_a, "energy_max"_a, "log_dos"_a = nb::none())
    .def(nb::init<const MolecularCluster &, const ClusterSpec &, FragmentationDosPtr, FragmentationDosPtr, FragmentationDosPtr>(),
         "daughter"_a, "neutral"_a, "daughter_dos"_a, "neutral_dos"_a.none(), "combined_dos"_a)
    .def_prop_ro("log_dos", &FragmentationProducts::log_dos)
    .def_prop_ro("daughter_dos", &FragmentationProducts::daughter_dos)
    .def_prop_ro("neutral_dos", &FragmentationProducts::neutral_dos, nb::sig("def neutral_dos(self) -> FragmentationDos | None"))
    .def_prop_ro("combined_dos", &FragmentationProducts::combined_dos);
  nb::class_<MassSpecInputFragmentationPathway>(m, "MassSpecInputFragmentationPathway")
    .def(nb::init<const MolecularCluster &, const MolecularCluster &, const ClusterSpec &, const Histogram &, std::optional<double>, FragmentationProductsPtr>(),
         "cluster_0"_a,
         "cluster_1"_a,
         "cluster_2"_a,
         "rate_const"_a,
         "bonding_energy"_a = std::nullopt, "products"_a = nullptr)
    .def(nb::init<Histogram, double, FragmentationProductsPtr>(), "rate_const"_a, "bonding_energy"_a = std::nullopt, "products"_a = nullptr)
    .def_ro("rate_const", &MassSpecInputFragmentationPathway::rate_const)
    .def_ro("bonding_energy", &MassSpecInputFragmentationPathway::bonding_energy)
    .def_ro("products", &MassSpecInputFragmentationPathway::products, nb::sig("def products(self) -> FragmentationProducts | None"));

  nb_magic_enum<VibEnergyMode>(m, "VibEnergyMode");
  nb::class_<RFBasis>(m, "RFBasis")
    .def(nb::init<double, double, int>(), "ac_scaled"_a, "dc_scaled"_a, "cells"_a = RFBasis::default_cells)
    .def(nb::init<double, double, const Eigen::ArrayXd &>(), "ac_scaled"_a, "dc_scaled"_a, "data"_a)
    .def_prop_ro("data", &RFBasis::data)
    .def_prop_ro_static("default_cells", [](nb::handle)
  { return RFBasis::default_cells; })
    .def_prop_ro_static("format_version", [](nb::handle)
  { return RFBasis::format_version; });
  m.def("rf_parameters", [](Quadrupole q, double m_ion)
  {
    q.compute_mathieu_factor(m_ion);
    return rf_parameters(q);
  }, "quadrupole"_a, "m_ion"_a);
  nb::class_<VibEnergyIndex>(m, "VibEnergyIndex")
    .def(nb::init<VibEnergyMode, Eigen::ArrayXd>(), "mode"_a, "data"_a);
  nb::class_<MassSpecSubstanceSingleInput>(m, "MassSpecSubstanceSingleInput")
    .def(nb::init<const MolecularCluster &, const MolecularCluster &, const ClusterSpec &, Gas, const Histogram &, const Histogram &, std::optional<double>, int, VibIndexPtr, RFBasisPtr>(),
         "cluster_0"_a,
         "cluster_1"_a,
         "cluster_2"_a,
         "gas"_a,
         "density_cluster"_a,
         "rate_const"_a,
         "fragmentation_energy"_a = std::nullopt,
         "cluster_charge_sign"_a = defaults::cluster_charge_sign, "vib_index"_a = nullptr, "rf_basis"_a = nullptr)
    .def(nb::init<const MolecularCluster &, std::vector<MassSpecInputFragmentationPathway>, Gas, const Histogram &, int, VibIndexPtr, RFBasisPtr>(),
         "cluster_0"_a,
         "pathways"_a,
         "gas"_a,
         "density_cluster"_a,
         "cluster_charge_sign"_a = defaults::cluster_charge_sign, "vib_index"_a = nullptr, "rf_basis"_a = nullptr)
    .def(nb::init<int, double, double, const Histogram, const std::vector<MassSpecInputFragmentationPathway>, const Gas, VibIndexPtr, RFBasisPtr>(),
         "cluster_charge_sign"_a,
         "m_ion"_a,
         "R_cluster"_a,
         "density_cluster"_a,
         "pathways"_a,
         "gas"_a, "vib_index"_a = nullptr, "rf_basis"_a = nullptr)
    .def_ro("cluster_charge_sign", &MassSpecSubstanceSingleInput::cluster_charge_sign)
    .def_ro("m_ion", &MassSpecSubstanceSingleInput::m_ion)
    .def_ro("R_cluster", &MassSpecSubstanceSingleInput::R_cluster)
    .def_ro("density_cluster", &MassSpecSubstanceSingleInput::density_cluster)
    .def_ro("pathways", &MassSpecSubstanceSingleInput::pathways)
    .def_ro("gas", &MassSpecSubstanceSingleInput::gas)
    .def_ro("rf_basis", &MassSpecSubstanceSingleInput::rf_basis, nb::sig("def rf_basis(self) -> RFBasis | None"));

  tree_bindings::bind_vector_view<size_t>(m, "IndexView");
  tree_bindings::bind_vector_view<MSSubstanceTreeNode>(m, "TreeNodeView");
  tree_bindings::bind_vector_view<MSSubstanceTreePathway>(m, "TreePathwayView");
  tree_bindings::bind_vector_view<MSSubstanceTreeCluster>(m, "TreeClusterView");
  tree_bindings::bind_vector_view<MassSpecInputFragmentationPathway>(m, "FragmentationPathwayView");

  nb::class_<MSSubstanceTreeCluster>(m, "MSSubstanceTreeCluster")
    .def(nb::init<double, double, Histogram, VibIndexPtr, RFBasisPtr>(),
         "m_ion"_a,
         "R_cluster"_a,
         "density_cluster"_a, "vib_index"_a = nullptr, "rf_basis"_a = nullptr)
    .def(nb::init<const MolecularCluster &, Histogram, VibIndexPtr, RFBasisPtr>(),
         "cluster_0"_a,
         "density_cluster"_a, "vib_index"_a = nullptr, "rf_basis"_a = nullptr)
    .def_ro("m_ion", &MSSubstanceTreeCluster::m_ion)
    .def_ro("R_cluster", &MSSubstanceTreeCluster::R_cluster)
    .def_ro("density_cluster", &MSSubstanceTreeCluster::density_cluster)
    .def_ro("rf_basis", &MSSubstanceTreeCluster::rf_basis, nb::sig("def rf_basis(self) -> RFBasis | None"));

  nb::class_<MSSubstanceTreeNode>(m, "MSSubstanceTreeNode")
    .def(nb::init<size_t, std::vector<size_t>>(),
         "payload_idx"_a,
         "pathway_indices"_a)
    .def_ro("payload_idx", &MSSubstanceTreeNode::payload_idx)
    .def_prop_ro("pathway_indices", tree_bindings::vector_property(&MSSubstanceTreeNode::pathway_indices));

  nb::class_<MSSubstanceTreePathway>(m, "MSSubstanceTreePathway")
    .def(nb::init<size_t, std::optional<size_t>>(),
         "payload_idx"_a,
         "product_idx"_a = nb::none())
    .def_ro("payload_idx", &MSSubstanceTreePathway::payload_idx)
    .def_ro("product_idx", &MSSubstanceTreePathway::product_idx);

  nb::class_<MassSpecSubstanceTreeInput>(m, "MassSpecSubstanceTreeInput")
    .def(nb::init<int, Gas, std::vector<MSSubstanceTreeCluster>, std::vector<MassSpecInputFragmentationPathway>, std::vector<MSSubstanceTreeNode>, std::vector<MSSubstanceTreePathway>, FragmentationStateMethod>(),
         "cluster_charge_sign"_a,
         "gas"_a,
         "cluster_payloads"_a,
         "pathway_payloads"_a,
         "tree_nodes"_a,
         "tree_pathways"_a, "fragmentation_state_method"_a = FragmentationStateMethod::zero)
    .def_ro("fragmentation_state_method", &MassSpecSubstanceTreeInput::fragmentation_state_method)
    .def_ro("cluster_charge_sign", &MassSpecSubstanceTreeInput::cluster_charge_sign)
    .def_ro("gas", &MassSpecSubstanceTreeInput::gas)
    .def_prop_ro("cluster_payloads", tree_bindings::vector_property(&MassSpecSubstanceTreeInput::cluster_payloads))
    .def_prop_ro("pathway_payloads", tree_bindings::vector_property(&MassSpecSubstanceTreeInput::pathway_payloads))
    .def_prop_ro("tree_nodes", tree_bindings::vector_property(&MassSpecSubstanceTreeInput::tree_nodes))
    .def_prop_ro("tree_pathways", tree_bindings::vector_property(&MassSpecSubstanceTreeInput::tree_pathways))
    .def("topology_columns", &tree_bindings::topology_columns,
         "Return owned int64 topology columns; destination -1 denotes a terminal transition.");

  m.def("validate_max_energies", static_cast<void (*)(double, double, double, double)>(validate_max_energies),
        "fragmentation_energy"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a);

  m.def("densityandrate",
        &densityandrate,
        nb::call_guard<nb::gil_scoped_release>(),
        "cluster_0"_a,
        "cluster_1"_a,
        "cluster_2"_a,
        "energy_max"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "fragmentation_energy"_a,
        "k_rate_cancellation"_a = true);

  m.def("compute_density_of_states_batch",
        &compute_density_of_states_batch,
        nb::call_guard<nb::gil_scoped_release>(),
        "batch_frequencies"_a,
        "energy_max"_a,
        "bin_width"_a,
        "use_old_impl"_a = false);

  nb::class_<KTotalInput>(m, "KTotalInput")
    .def(nb::init<const MolecularCluster &, const ClusterSpec &, double, Eigen::Ref<const Eigen::ArrayXd>, Eigen::Ref<const Eigen::ArrayXd>>(),
         nb::arg("cluster_1"),
         nb::arg("cluster_2"),
         nb::arg("fragmentation_energy"),
         nb::arg("rho_parent"),
         nb::arg("rho_comb"))
    .def_ro("cluster_1", &KTotalInput::cluster_1)
    .def_ro("cluster_2", &KTotalInput::cluster_2)
    .def_ro("fragmentation_energy", &KTotalInput::fragmentation_energy)
    .def_ro("rho_parent", &KTotalInput::rho_parent)
    .def_ro("rho_comb", &KTotalInput::rho_comb);

  m.def("precompute_mesh",
        &precompute_mesh,
        nb::call_guard<nb::gil_scoped_release>(),
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh_mode"_a);

  m.def("compute_k_total_batch",
        static_cast<Eigen::ArrayXXd (*)(const std::vector<KTotalInput> &, double, double, MeshMode, std::optional<std::function<void(size_t)>>, bool)>(compute_k_total_batch),
        nb::call_guard<nb::gil_scoped_release>(),
        "batch_input"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh_mode"_a,
        "progress_callback"_a = std::nullopt,
        "k_rate_cancellation"_a = true);

  m.def("compute_k_total_batch",
        static_cast<Eigen::ArrayXXd (*)(const std::vector<KTotalInput> &, double, double, const std::optional<Eigen::ArrayXd> &, std::optional<std::function<void(size_t)>>, bool)>(compute_k_total_batch),
        nb::call_guard<nb::gil_scoped_release>(),
        "batch_input"_a,
        "energy_max_rate"_a,
        "bin_width"_a,
        "mesh"_a = std::nullopt,
        "progress_callback"_a = std::nullopt,
        "k_rate_cancellation"_a = true);

  nb_magic_enum<SampleMode>(m, "SampleMode");
  nb_magic_enum<CollisionTimeMethod>(m, "CollisionTimeMethod");
  const CollisionTimeOptions default_collision_time;
  nb::class_<CollisionTimeOptions>(m, "CollisionTimeOptions")
    .def(nb::init<CollisionTimeMethod, double, double>(), "method"_a = default_collision_time.method,
         "hazard_atol"_a = default_collision_time.hazard_atol, "hazard_rtol"_a = default_collision_time.hazard_rtol)
    .def_ro("method", &CollisionTimeOptions::method)
    .def_ro("hazard_atol", &CollisionTimeOptions::hazard_atol)
    .def_ro("hazard_rtol", &CollisionTimeOptions::hazard_rtol);

  m.def("scaled_density",
        &scaled_density,
        "density_cluster"_a);

  m.def("scaled_rate_const",
        &scaled_rate_const,
        "scaled_rate_const"_a);

  m.def("mass_spec",
        &mass_spec<MassSpecSubstanceSingleInput>,
        nb::call_guard<nb::gil_scoped_release>(),
        "ms"_a,
        "subs"_a,
        "N"_a,
        "seed"_a = DEFAULT_SEED,
        "log_callback"_a = std::nullopt,
        "result_callback"_a = std::nullopt,
        "sample_mode"_a = SampleMode::rejection,
        "strict"_a = true,
        "logconf"_a = DEFAULT_LOGCONF_TUPLE,
        "collision_time"_a = std::nullopt,
        "ordered"_a = false);

  m.def("mass_spec",
        &mass_spec<MassSpecSubstanceTreeInput>,
        nb::call_guard<nb::gil_scoped_release>(),
        "ms"_a,
        "subs"_a,
        "N"_a,
        "seed"_a = DEFAULT_SEED,
        "log_callback"_a = std::nullopt,
        "result_callback"_a = std::nullopt,
        "sample_mode"_a = SampleMode::rejection,
        "strict"_a = true,
        "logconf"_a = DEFAULT_LOGCONF_TUPLE,
        "collision_time"_a = std::nullopt,
        "ordered"_a = false);

  nb::class_<ParticleStateMsg>(m, "ParticleState")
    .def_ro("realization", &ParticleStateMsg::realization)
    .def_prop_ro("postime", &ParticleStateMsg::postime)
    .def_prop_ro("velocity", [](const ParticleStateMsg &state) -> const Eigen::Vector3d &
  {
    return state.particle.velocity;
  }, nb::rv_policy::reference_internal)
    .def_prop_ro("omega", [](const ParticleStateMsg &state) -> const Eigen::Vector3d &
  {
    return state.particle.omega;
  }, nb::rv_policy::reference_internal)
    .def_ro("rot_energy", &ParticleStateMsg::rot_energy)
    .def_prop_ro("vibrational_energy", [](const ParticleStateMsg &state)
  {
    return state.particle.vibrational_energy;
  }).def_ro("particle_index", &ParticleStateMsg::particle_index);

  nb::class_<InitEvent>(m, "InitEvent")
    .def_ro("state", &InitEvent::state);

  nb::class_<CollisionEvent>(m, "CollisionEvent")
    .def_ro("state", &CollisionEvent::state)
    .def_ro("theta", &CollisionEvent::theta)
    .def_ro("u_norm", &CollisionEvent::u_norm)
    .def_ro("accepted", &CollisionEvent::accepted);

  nb::class_<PostCollisionEvent>(m, "PostCollisionEvent")
    .def_ro("state", &PostCollisionEvent::state);

  nb::class_<FragmentationEvent>(m, "FragmentationEvent")
    .def_ro("state", &FragmentationEvent::state)
    .def_ro("pathway_index", &FragmentationEvent::pathway_index)
    .def_ro("next_particle_index", &FragmentationEvent::next_particle_index);

  nb::class_<PostFragmentationEvent>(m, "PostFragmentationEvent")
    .def_ro("state", &PostFragmentationEvent::state);

  nb::class_<EscapeEvent>(m, "EscapeEvent")
    .def_ro("state", &EscapeEvent::state);

  nb::class_<MassSpecBatch>(m, "MassSpecBatch")
    .def_prop_ro("columns", [](const MassSpecBatch &batch)
  { return event_batch::export_columns(batch.events); })
    .def_ro("items", &MassSpecBatch::items);

  nb::exception<IpcWriterBusy>(m, "IpcWriterBusy", PyExc_RuntimeError);
  nb::class_<IpcEventWriter>(m, "IpcEventWriter")
    .def(nb::init<const std::string &, bool, const std::string &>(), "path"_a, "append"_a, "archive_id"_a)
    .def(nb::init<const std::string &, const std::string &, int64_t, bool, int>(),
         "path"_a, "archive_id"_a, "generation"_a, "zstd"_a, "column_workers"_a)
    .def("close", &IpcEventWriter::close, "metadata"_a = std::nullopt, nb::call_guard<nb::gil_scoped_release>())
    .def("abort", &IpcEventWriter::abort, nb::call_guard<nb::gil_scoped_release>())
    .def("set_metadata", &IpcEventWriter::set_metadata, "metadata"_a)
    .def_prop_ro("batch_count", &IpcEventWriter::batch_count)
    .def_prop_ro("rows_written", &IpcEventWriter::rows_written);

  nb::class_<StreamDelivery>(m, "StreamDelivery")
    .def(nb::init<std::shared_ptr<IpcEventWriter>, int64_t, bool, bool, bool, bool>(),
         "writer"_a.none(), "result_id"_a, "events"_a, "ordered"_a, "progress"_a, "logs"_a);

  nb::class_<MassSpecIterator>(m, "MassSpecIterator")
    .def(nb::init<std::shared_ptr<const MassSpectrometer>, std::shared_ptr<const MassSpecSubstanceSingleInput>, int, unsigned long long, SampleMode, bool, std::tuple<int, bool>, std::optional<CollisionTimeOptions>, size_t, size_t, size_t, std::optional<StreamDelivery>>(),
         nb::call_guard<nb::gil_scoped_release>(),
         "ms"_a,
         "subs"_a,
         "N"_a,
         "seed"_a = DEFAULT_SEED,
         "sample_mode"_a = SampleMode::rejection,
         "strict"_a = true,
         "logconf"_a = DEFAULT_LOGCONF_TUPLE,
         "collision_time"_a = std::nullopt,
         "initial_batch_size"_a = 4096,
         "max_batch_size"_a = 262144,
         "queue_capacity"_a = 0,
         "stream"_a = std::nullopt)
    .def(nb::init<std::shared_ptr<const MassSpectrometer>, std::shared_ptr<const MassSpecSubstanceTreeInput>, int, unsigned long long, SampleMode, bool, std::tuple<int, bool>, std::optional<CollisionTimeOptions>, size_t, size_t, size_t, std::optional<StreamDelivery>>(),
         nb::call_guard<nb::gil_scoped_release>(),
         "ms"_a,
         "subs"_a,
         "N"_a,
         "seed"_a = DEFAULT_SEED,
         "sample_mode"_a = SampleMode::rejection,
         "strict"_a = true,
         "logconf"_a = DEFAULT_LOGCONF_TUPLE,
         "collision_time"_a = std::nullopt,
         "initial_batch_size"_a = 4096,
         "max_batch_size"_a = 262144,
         "queue_capacity"_a = 0,
         "stream"_a = std::nullopt)
    .def("__next__", &MassSpecIterator::__next__)
    .def("read_batch", &MassSpecIterator::read_batch, "max_items"_a = 262144)
    .def("join_if_joinable", &MassSpecIterator::join_if_joinable, nb::call_guard<nb::gil_scoped_release>())
    .def("request_cancel", &MassSpecIterator::request_cancel)
    .def("close", &MassSpecIterator::close, nb::call_guard<nb::gil_scoped_release>())
    .def("run_result", &MassSpecIterator::run_result)
    .def_prop_ro("recording_range", [](MassSpecIterator &stream)
  {
    stream.require_stopped();
    return stream.recording_range;
  }).def_prop_ro("cancelled", [](const MassSpecIterator &stream)
  { return stream.cancelled.load(); })
    .def_prop_ro("partial_counters", [](MassSpecIterator &stream)
  {
    if (stream.has_pump)
      stream.require_stopped();
    return Eigen::ArrayXi(stream.partial_counters);
  });

  nb::class_<FragmentationPathway>(m, "FragmentationPathway")
    .def(nb::init<const MolecularCluster &, const MolecularCluster &, const ClusterSpec &>(),
         nb::arg("parent"),
         nb::arg("product1"),
         nb::arg("product2"))
    .def("fragmentation_energy_kelvin", &FragmentationPathway::fragmentation_energy_kelvin);

  nb_magic_enum<Counter::Counter>(m, "Counter");
  nb_magic_enum<MeshMode>(m, "MeshMode");

  nb::exception<ApiTofError>(m, "ApiTofError");
  nb::exception<ApiTofArgumentError>(m, "ApiTofArgumentError", m.attr("ApiTofError"));
  nb::exception<ApiTofOverflowError>(m, "ApiTofOverflowError", m.attr("ApiTofError"));
  nb::exception<ApiTofDosOverflow> PyApiTofDosOverflow(m, "ApiTofDosOverflow", m.attr("ApiTofOverflowError"));
  nb::exception<ApiTofRateConstantOverflow> PyApiTofRateConstantOverflow(m, "ApiTofRateConstantOverflow", m.attr("ApiTofOverflowError"));
  nb::exception<ApiTofMaxCollisions> PyApiTofMaxCollisions(m, "ApiTofMaxCollisions", m.attr("ApiTofOverflowError"));
  nb::exception<ApiTofParticleEscapeError>(m, "ApiTofParticleEscapeError", m.attr("ApiTofError"));
  nb::exception<ApiTofUnexpectedNumericalError>(m, "ApiTofUnexpectedNumericalError", m.attr("ApiTofError"));

  register_overflow_translator<ApiTofDosOverflow>(PyApiTofDosOverflow);
  register_overflow_translator<ApiTofRateConstantOverflow>(PyApiTofRateConstantOverflow);
  register_overflow_translator<ApiTofMaxCollisions>(PyApiTofMaxCollisions);

  m.def("sample_collision", &sample_collision,
        "sample_mode"_a,
        "num_samples"_a,
        "v_rel_norm"_a,
        "gas"_a,
        "R_cluster"_a,
        "P"_a,
        "T"_a,
        "seed"_a = 42,
        "dtheta"_a = 1.0e-3,
        "du"_a = std::nullopt);

  m.def("debug_info", &debug_info);
  m.attr("DEFAULT_LOGLEVEL") = DEFAULT_LOGLEVEL;

  nb::module_ m_defaults = m.def_submodule("defaults", "Default parameter values");

  m_defaults.attr("cluster_charge_sign") = defaults::cluster_charge_sign;

  nb::module_ m_consts = m.def_submodule("consts", "Constants");
  m_consts.attr("hartK") = consts::hartK;
  m_consts.attr("boltzmann") = consts::boltzmann;
}
