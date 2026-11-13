#pragma once

#include "event_columns.h"
#include <memory>
#include <variant>

// Immutable after dispatch. Shared storage is used only for Parquet + Python.
class EventBatch
{
  std::variant<std::unique_ptr<EventColumns>, std::shared_ptr<const EventColumns>> storage;

public:
  EventBatch() = default;
  EventBatch(std::unique_ptr<EventColumns> columns) : storage(std::move(columns))
  {
  }
  EventBatch(std::shared_ptr<const EventColumns> columns) : storage(std::move(columns))
  {
  }
  [[nodiscard]] const EventColumns *get() const
  {
    return std::visit([](const auto &columns) -> const EventColumns *
    { return columns.get(); }, storage);
  }
  const EventColumns *operator->() const
  {
    return get();
  }
  // Raw iterator reuse is allowed only after all Python views release their owner.
  std::unique_ptr<EventColumns> release_unique() &&
  {
    return std::get<std::unique_ptr<EventColumns>>(std::move(storage));
  }
};
