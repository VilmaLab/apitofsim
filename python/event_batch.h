#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "event_columns.h"

namespace event_batch
{
namespace nb = nanobind;

using Columns = EventColumns;

inline nb::typed<nb::dict, nb::str, nb::object> export_columns(const std::shared_ptr<Columns> &columns)
{
  nb::capsule owner(new std::shared_ptr<Columns>(columns), [](void *pointer) noexcept
  { delete static_cast<std::shared_ptr<Columns> *>(pointer); });
  constexpr std::array names{"x", "y", "z", "t", "vx", "vy", "vz", "ox", "oy", "oz", "rot_energy", "vibrational_energy", "theta", "u_norm"};
  constexpr std::array index_names{"realization", "event_kind", "particle_index", "transition_index", "next_particle_index", "accepted"};
  nb::typed<nb::dict, nb::str, nb::object> out;
  auto add = [&]<typename T>(const char *name, const std::vector<T> &data)
  {
    static const T empty{};
    out[name] = nb::ndarray<nb::numpy, const T, nb::ndim<1>>(data.empty() ? &empty : data.data(), {data.size()}, owner);
  };
  for (size_t i = 0; i < names.size(); ++i)
    add(names[i], columns->values[i]);
  for (size_t i = 0; i < index_names.size(); ++i)
    add(index_names[i], columns->indices[i]);
  if (!columns->event_indices.empty())
  {
    add("result_id", columns->result_ids);
    add("event_index", columns->event_indices);
  }
  return out;
}
} // namespace event_batch
