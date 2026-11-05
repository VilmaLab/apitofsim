#pragma once

#include <array>
#include <cstdint>
#include <vector>
#include "warnlogcount.h"

// Columns stay owned by the returned arrays. A retained Python/Arrow batch must
// never be overwritten, even after its iterator has advanced or been destroyed.
struct EventColumns
{
  std::vector<int64_t> result_ids;
  std::vector<int64_t> event_indices;
  std::array<std::vector<double>, 14> values;
  std::array<std::vector<int64_t>, 6> indices;

  void clear(size_t capacity)
  {
    auto reset = [capacity](auto &columns)
    {
      for (auto &column : columns)
      {
        column.clear();
        column.reserve(capacity);
      }
    };
    result_ids.clear();
    event_indices.clear();
    reset(values);
    reset(indices);
  }

  void append(const EventMessage &message)
  {
    std::visit([&](const auto &event)
    {
      const auto &state = event.state;
      const auto &particle = state.particle;
      std::array<double, 14> row{particle.position.x(), particle.position.y(), particle.position.z(), particle.time,
                                 particle.velocity.x(), particle.velocity.y(), particle.velocity.z(),
                                 particle.omega.x(), particle.omega.y(), particle.omega.z(),
                                 state.rot_energy, particle.vibrational_energy, 0, 0};
      std::array<int64_t, 6> ids{state.realization, static_cast<int64_t>(message.index()), state.particle_index, -1, -1, 0};
      using Event = std::decay_t<decltype(event)>;
      if constexpr (std::is_same_v<Event, CollisionEvent>)
      {
        row[12] = event.theta;
        row[13] = event.u_norm;
        ids[5] = event.accepted;
      }
      else if constexpr (std::is_same_v<Event, FragmentationEvent>)
      {
        ids[3] = event.pathway_index;
        ids[4] = event.next_particle_index;
      }
      for (size_t i = 0; i < row.size(); ++i)
        values[i].push_back(row[i]);
      for (size_t i = 0; i < ids.size(); ++i)
        indices[i].push_back(ids[i]);
    }, message);
  }
};
