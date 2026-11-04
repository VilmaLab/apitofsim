from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from apitofsim import apitofsimraw as raw


def molecular(rotations=(1.0, 2.0, 3.0)):
    return raw.MolecularCluster(10, 0.0, np.array(rotations), np.array([2.0]), 0)


def prefactor(first, second):
    c = raw.consts
    hbar = 1.054571800e-34  # The value used in src/include/consts.h.
    inertia = raw.compute_derived(first).inertia_moment
    if isinstance(second, raw.AtomLikeCluster):
        return c.boltzmann**2 * inertia / (np.pi * hbar**3)
    inertia += raw.compute_derived(second).inertia_moment
    rotations = np.prod(first.rotations) ** (1 / 3) + np.prod(second.rotations) ** (
        1 / 3
    )
    return 2 * c.boltzmann**2 * inertia / (np.pi * hbar**3 * rotations**1.5)


def reference(first, second, parent, combined, shift, h, size, mesh=None):
    """Original two-stage midpoint sums, including the intermediate division."""
    rotations = np.sqrt(h * (np.arange(len(parent)) + 0.5))
    translations = h * (np.arange(size) + 0.5)
    atom = isinstance(second, raw.AtomLikeCluster)
    if atom:
        kernel = translations
    else:
        kernel = (
            np.convolve(translations, rotations[:size])[:size] if mesh is None else mesh
        )
    inner = np.convolve(combined[:size], kernel)[:size]
    k0 = prefactor(first, second) / parent[shift : shift + size] * inner
    k0 *= h if atom else h**2
    numerator = np.convolve(parent[shift : shift + size] * k0, rotations[:size])[:size]
    denominator = np.convolve(parent, rotations)[shift : shift + size]
    return numerator / denominator


@pytest.mark.parametrize("h", [0.5, 4.0])
@pytest.mark.parametrize("custom_mesh", [False, True])
@pytest.mark.parametrize("cancellation", [True, False])
def test_shared_rates_with_overlapping_and_disjoint_windows(
    h, custom_mesh, cancellation
):
    size = 80
    parent = np.linspace(1, 9, size + 200) ** 8
    combined = np.linspace(0, 8, size + 10) ** 6
    combined[:3] = 0
    first = molecular()
    seconds = [molecular(), raw.AtomLikeCluster(1, 0.0, 0), molecular((2.0, 3.0, 4.0))]
    shifts = [30, 0, 50, 30, 200]
    mesh = np.linspace(0.5, 100.0, size) if custom_mesh else None
    inputs = [
        raw.KTotalInput(
            first,
            seconds[i % 3],
            (shift + 0.25) * h,
            parent[: shift + size],
            combined[:size] if i % 2 else combined,
        )
        for i, shift in enumerate(shifts)
    ]
    expected = np.column_stack(
        [
            reference(first, seconds[i % 3], parent, combined, shift, h, size, mesh)
            for i, shift in enumerate(shifts)
        ]
    )
    mode = mesh if custom_mesh else raw.MeshMode.compute_mesh_linear
    completed = []
    actual = raw.compute_k_total_batch(
        inputs, size * h, h, mode, completed.append, k_rate_cancellation=cancellation
    )
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)
    assert completed == list(range(1, len(inputs) + 1))
    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(
                raw.compute_k_total_batch,
                inputs[::-1],
                size * h,
                h,
                mode,
                k_rate_cancellation=cancellation,
            )
            for _ in range(2)
        ]
        for future in futures:
            np.testing.assert_allclose(
                future.result(), expected[:, ::-1], rtol=2e-12, atol=0
            )
    # Caches belong to one call and must not retain values after caller mutation.
    combined *= 2
    again = raw.compute_k_total_batch(
        inputs, size * h, h, mode, k_rate_cancellation=cancellation
    )
    np.testing.assert_allclose(again, 2 * expected, rtol=2e-12, atol=0)


@pytest.mark.parametrize("atom", [False, True])
def test_cancelled_rates_preserve_large_dynamic_ranges(atom):
    size, shift, h = 128, 37, 4.0
    parent = np.geomspace(1, 1e180, size + shift)
    combined = np.geomspace(1, 1e160, size)
    first = molecular()
    second = raw.AtomLikeCluster(1, 0.0, 0) if atom else molecular()
    inputs = [raw.KTotalInput(first, second, shift * h, parent, combined)]
    actual = raw.compute_k_total_batch(
        inputs, size * h, h, raw.MeshMode.compute_mesh_linear
    )[:, 0]
    expected = reference(first, second, parent, combined, shift, h, size)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=0)


def test_cancelled_rate_handles_isolated_zero_parent_bins():
    first = molecular()
    parent = np.array([1.0, 0.0, 1.0])
    combined = np.array([1.0, 0.0, 0.0])
    inputs = [raw.KTotalInput(first, first, 0.0, parent, combined)]
    rates = raw.compute_k_total_batch(
        inputs, 3.0, 1.0, raw.MeshMode.compute_mesh_linear
    )[:, 0]
    assert np.all(np.isfinite(rates))
    assert rates[1] == pytest.approx(
        prefactor(first, first) * (0.75 + np.sqrt(0.75)) / np.sqrt(1.5)
    )
    parent[:] = 0
    with pytest.raises(raw.ApiTofArgumentError, match="normalization"):
        raw.compute_k_total_batch(inputs, 3.0, 1.0, raw.MeshMode.compute_mesh_linear)


def test_cancelled_rate_validates_density_and_mesh_lengths():
    first = molecular()
    parent = np.ones(8)
    combined = np.ones(4)
    inputs = [raw.KTotalInput(first, first, 0.0, parent, combined)]
    with pytest.raises(raw.ApiTofArgumentError, match="Product DOS"):
        raw.compute_k_total_batch(inputs, 5.0, 1.0, raw.MeshMode.compute_mesh_linear)
    with pytest.raises(raw.ApiTofArgumentError, match="Mesh"):
        raw.compute_k_total_batch(inputs, 4.0, 1.0, np.ones(3))
    for energy in (-1.0, float("nan"), float("inf"), 1e100):
        invalid = [raw.KTotalInput(first, first, energy, parent, combined)]
        with pytest.raises(raw.ApiTofArgumentError):
            raw.compute_k_total_batch(
                invalid, 4.0, 1.0, raw.MeshMode.compute_mesh_linear
            )


@pytest.mark.parametrize("atom", [False, True])
def test_disabling_cancellation_preserves_two_stage_zero_bin_behavior(atom):
    from apitofsim import ClusterData, compute_k_total_batch, densityandrate
    from apitofsim.api import Q_

    first = molecular()
    second = raw.AtomLikeCluster(1, 0.0, 0) if atom else molecular()
    parent = np.array([1.0, 0.0, 1.0])
    combined = np.array([1.0, 0.0, 1.0])
    inputs = [raw.KTotalInput(first, second, 0.0, parent, combined)]
    for cancellation in (True, False):
        rates = compute_k_total_batch(
            inputs, Q_(3, "K"), Q_(1, "K"), k_rate_cancellation=cancellation
        )
        assert np.isfinite(rates[1, 0]) == cancellation
        cluster = ClusterData(
            Q_(10, "amu"),
            Q_(0, "hartree"),
            np.array([1.0, 2.0, 3.0]),
            np.array([2.0]),
            0,
        )
        product = (
            ClusterData(Q_(1, "amu"), Q_(0, "hartree"), np.zeros(3), np.array([]), 0)
            if atom
            else cluster
        )
        _, histogram = densityandrate(
            cluster,
            cluster,
            product,
            Q_(5, "K"),
            Q_(3, "K"),
            Q_(1, "K"),
            fragmentation_energy=Q_(2, "K"),
            k_rate_cancellation=cancellation,
        )
        assert np.isfinite(histogram.y[1]) == cancellation
    default = compute_k_total_batch(inputs, Q_(3, "K"), Q_(1, "K"))
    assert np.isfinite(default).all()
