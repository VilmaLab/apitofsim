#pragma once

#include "event_columns.h"
#include <memory>

// Immutable after dispatch. Shared storage is used only for Parquet + Python.
class EventBatch
{
  std::unique_ptr<EventColumns> owned;
  std::shared_ptr<const EventColumns> shared;

public:
  EventBatch() = default;
  EventBatch(std::unique_ptr<EventColumns> columns) : owned(std::move(columns))
  {
  }
  EventBatch(std::shared_ptr<const EventColumns> columns) : shared(std::move(columns))
  {
  }
  [[nodiscard]] const EventColumns *get() const
  {
    return owned ? owned.get() : shared.get();
  }
  const EventColumns *operator->() const
  {
    return get();
  }
  // Raw iterator reuse is allowed only after all Python views release their owner.
  std::unique_ptr<EventColumns> release_unique() &&
  {
    return std::move(owned);
  }
};
