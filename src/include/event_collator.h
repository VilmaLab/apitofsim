#pragma once

#include "arrow_event_writer.h"
#include "event_recording.h"
#include "parquet_event_writer.h"
#include <algorithm>
#include <functional>
#include <stdexcept>

struct NullWriter
{
  static constexpr bool enabled = false;
  static constexpr bool retains_batch = false;
};
struct NullCallbackArrow
{
  static constexpr bool enabled = false;
};
struct CallbackArrow
{
  static constexpr bool enabled = true;
  std::function<void(EventBatch)> deliver;
  void operator()(EventBatch batch) const
  {
    deliver(std::move(batch));
  }
};

// Only dispatch is templated. Collation and the simulation loop are shared.
template <typename Writer, typename Callback>
struct EventDispatch
{
  EventWriter::Lease *writer;
  [[no_unique_address]] Callback callback;
  void operator()(std::unique_ptr<EventColumns> &columns) const
  {
    if constexpr (Writer::enabled && !Writer::retains_batch)
      writer->write(*columns);
    if constexpr (Writer::retains_batch && Callback::enabled)
    {
      std::shared_ptr<const EventColumns> shared(std::move(columns));
      writer->write(EventBatch(shared));
      callback(EventBatch(std::move(shared)));
    }
    else if constexpr (Writer::retains_batch)
    {
      writer->write(EventBatch(std::move(columns)));
    }
    else if constexpr (Callback::enabled)
    {
      callback(EventBatch(std::move(columns)));
    }
  }
};

template <typename Writer, typename Callback>
auto event_dispatch(EventWriter::Lease *writer, Callback callback)
{
  return EventDispatch<Writer, Callback>{writer, std::move(callback)};
}

class EventCollator
{
  using Dispatch = std::function<void(std::unique_ptr<EventColumns> &)>;
  Dispatch dispatch;
  std::unique_ptr<EventColumns> columns;
  std::vector<int64_t> ordinals;
  int64_t result_id;
  size_t limit;
  size_t maximum;

public:
  EventCollator(bool enabled, int realizations, int64_t result_id,
                size_t initial, size_t maximum, Dispatch dispatch)
      : dispatch(std::move(dispatch)),
        ordinals(enabled ? static_cast<size_t>(std::max(0, realizations)) : 0, 0),
        result_id(result_id), limit(std::min(initial, maximum)), maximum(maximum)
  {
    if (enabled)
    {
      columns = std::make_unique<EventColumns>();
      columns->clear(limit);
    }
  }
  // Return true when the caller should flush events and pending control items.
  bool append(const EventMessage &event)
  {
    if (!columns)
      return false;
    columns->append(event);
    const auto realization = columns->indices[0].back();
    if (realization < 0 || static_cast<size_t>(realization) >= ordinals.size())
      throw std::runtime_error("Native event has an invalid realization index");
    columns->event_indices.push_back(ordinals[realization]++);
    return columns->event_indices.size() == limit;
  }
  void flush()
  {
    if (!columns || columns->event_indices.empty())
      return;
    columns->result_ids.assign(columns->event_indices.size(), result_id);
    dispatch(columns);
    limit = maximum;
    if (!columns)
      columns = std::make_unique<EventColumns>();
    columns->clear(limit);
  }
};
