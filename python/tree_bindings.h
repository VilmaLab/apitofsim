#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>

#include <nanobind/nanobind.h>
#include <nanobind/make_iterator.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>

#include "mass_spec.h"

namespace tree_bindings
{
namespace nb = nanobind;

// A distinct wrapper leaves list-to-vector constructor conversions intact.
// The owner chain is element -> view -> native tree (or containing node).
template <typename T>
struct VectorView
{
  const std::vector<T> *values;
  nb::object owner;
};

template <typename T>
void bind_vector_view(nb::module_ &m, const char *name)
{
  using View = VectorView<T>;
  nb::class_<View>(m, name)
    .def("__len__", [](const View &v)
  { return v.values->size(); })
    .def("__getitem__", [](const View &v, Py_ssize_t i) -> const T &
  {
    auto size = static_cast<Py_ssize_t>(v.values->size());
    if (i < 0)
      i += size;
    if (i < 0 || i >= size)
      throw nb::index_error();
    return (*v.values)[i];
  }, nb::rv_policy::reference_internal)
    .def("__getitem__", [](const View &v, const nb::slice &slice)
  {
    auto [start, stop, step, length] = slice.compute(v.values->size());
    (void)stop;
    nb::typed<nb::list, T> out;
    for (size_t i = 0; i < length; ++i, start += step)
      out.append(nb::cast((*v.values)[start], nb::rv_policy::copy));
    return out;
  }).def("__iter__", [](const View &v)
  {
    return nb::make_iterator<nb::rv_policy::reference_internal>(
      nb::type<View>(), "Iterator", v.values->cbegin(), v.values->cend());
  }, nb::keep_alive<0, 1>());
}

template <typename Owner, typename T>
auto vector_property(std::vector<T> Owner::*member)
{
  return [member](nb::handle owner)
  {
    return VectorView<T>{&(nb::cast<const Owner &>(owner).*member),
                         nb::borrow<nb::object>(owner)};
  };
}

using TopologyColumn = nb::ndarray<nb::numpy, int64_t, nb::ndim<1>>;
using TopologyColumns = nb::typed<nb::dict, nb::str, TopologyColumn>;

inline TopologyColumns topology_columns(const MassSpecSubstanceTreeInput &tree)
{
  auto column = [](size_t count)
  {
    auto *data = new int64_t[count];
    nb::capsule owner(data, [](void *p) noexcept
    { delete[] static_cast<int64_t *>(p); });
    return TopologyColumn(data, {count}, owner);
  };
  auto index = [](size_t value) -> int64_t
  {
    if (value > static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw nb::value_error("Tree payload index exceeds int64 range");
    return static_cast<int64_t>(value);
  };
  auto nodes = column(tree.tree_nodes.size());
  auto sources = column(tree.tree_pathways.size());
  auto payloads = column(tree.tree_pathways.size());
  auto destinations = column(tree.tree_pathways.size());
  std::fill_n(sources.data(), sources.size(), -1);
  for (size_t n = 0; n < tree.tree_nodes.size(); ++n)
  {
    nodes.data()[n] = index(tree.tree_nodes[n].payload_idx);
    for (size_t i : tree.tree_nodes[n].pathway_indices)
    {
      if (i >= sources.size() || sources.data()[i] != -1)
        throw nb::value_error("Tree transition must have exactly one source node");
      sources.data()[i] = n;
    }
  }
  for (size_t i = 0; i < tree.tree_pathways.size(); ++i)
  {
    if (sources.data()[i] == -1)
      throw nb::value_error("Tree transition has no source node");
    const auto &pathway = tree.tree_pathways[i];
    if (pathway.product_idx && *pathway.product_idx >= tree.tree_nodes.size())
      throw nb::value_error("Tree transition destination is out of range");
    payloads.data()[i] = index(pathway.payload_idx);
    destinations.data()[i] = pathway.product_idx ? static_cast<int64_t>(*pathway.product_idx) : -1;
  }
  TopologyColumns out;
  out["node_payload"] = nodes;
  out["transition_source"] = sources;
  out["transition_payload"] = payloads;
  out["transition_destination"] = destinations;
  return out;
}
} // namespace tree_bindings
