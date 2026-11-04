# /// script
# requires-python = ">=3.12"
# dependencies = ["marimo>=0.23.9", "numpy", "matplotlib", "duckdb"]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium")


@app.cell
def _():
    from pathlib import Path

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np

    return Path, duckdb, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Where does the collision energy go?

    A collision leaves a fixed amount of energy to share. How much should end up in the
    cluster's vibrations? Try **Rising DOS** first. Then switch to **Sparse DOS** or a
    Besel cluster such as **1A_neutral**. Watch where the proposed bins fall relative
    to the bins that survive rejection.

    ## One distribution for two physical processes

    Let $E$ be the available energy, $x_i$ the vibrational energy at the center of bin
    $i$, and $\rho_i$ its vibrational density of states (DOS). The energy left for
    the other motion is $r_i=E-x_i$. Energies on the controls and axes are expressed
    as energy divided by Boltzmann's constant, in kelvin. All bins have equal width.

    * **During a collision**, $E=x_{\rm old}+\mu u_n^2/2$, where $\mu$ is the reduced
      mass and $u_n$ the incoming normal relative speed. The residual goes to translation.
    * **After a collision**, $E$ is the sum of the vibrational and rotational energies.
      The residual goes to rotation.

    The paper gives both residual motions a DOS proportional to $\sqrt{r_i}$.
    Changing from residual to vibrational energy gives the same bin probabilities
    in both cases. With $(a)_+=\max(a,0)$,

    $$w_i(E)=\rho_i\sqrt{(E-x_i)_+},\qquad Z(E)=\sum_j w_j(E),\qquad
    p_i(E)=w_i(E)/Z(E).$$

    The bin width and unit-conversion factors cancel. When $Z(E)=0$, the simulator
    returns zero vibrational energy. Initialization is different: at temperature $T$ its
    bin weights are $\rho_i\exp[-x_i/(k_B T)]$ in physical energy units. Those weights
    permit a single reusable table at fixed $T$.

    [Zapadinsky et al., Eqs. 7, 8 and 16, pp. 614–615](https://doi.org/10.1021/acs.jpca.8b10744)
    """)
    return


@app.cell
def _(Path, duckdb, mo):
    database = Path(__file__).resolve().parents[1] / "besel2020/database.duckdb"
    curves = {"Rising DOS": -1, "Sparse DOS": -2, "Flat DOS": -3}
    if database.is_file():
        with duckdb.connect(str(database), read_only=True) as _db:
            for _id, _name in _db.execute("""
                select d.id, c.common_name from cluster_dos d
                join cluster c on c.id = d.cluster_id order by c.common_name, d.id
            """).fetchall():
                curves[f"{_name} (DOS {_id})"] = _id
    curve = mo.ui.dropdown(curves, value="Rising DOS", label="DOS curve")
    resolution = mo.ui.dropdown(
        {"250": 250, "2,500": 2500, "25,000": 25000},
        value="2,500",
        label="Bins on toy curves",
    )
    energy = mo.ui.slider(
        0, 100000, step=100, value=15000, label="E (K)", show_value=True
    )
    spare = mo.ui.slider(
        0,
        5000,
        step=100,
        value=500,
        label="Upper table headroom H − E (K)",
        show_value=True,
    )
    mo.vstack(
        [
            mo.md("## Choose a curve and an energy"),
            mo.hstack([curve, resolution], justify="start"),
            energy,
            spare,
            mo.md(
                "Besel curves use the stored grid. The toy grids all span 0–100,000 K."
                if database.is_file()
                else "Toy curves are available. To load Besel curves, place the notebook beside the examples’ besel2020 directory."
            ),
        ]
    )
    return curve, database, energy, resolution, spare


@app.cell
def _():
    return


@app.cell
def _(curve, database, duckdb, np, resolution):
    if curve.value >= 0:
        with duckdb.connect(str(database), read_only=True) as _db:
            _data, width = _db.execute(
                """
                select d.data, h.bin_width from cluster_dos d
                join histogram_params h on h.id = d.histogram_params_id where d.id = ?
            """,
                [curve.value],
            ).fetchone()
        rho = np.asarray(_data)
        x = width * (np.arange(len(rho)) + 0.5)
    else:
        width = 100000 / resolution.value
        x = width * (np.arange(resolution.value) + 0.5)
        rho = (1 + x / 1000) ** (0 if curve.value == -3 else 12)
        if curve.value == -2:
            rho = np.where(x % 2000 < 400, rho, 0)
    return rho, width, x


@app.cell(hide_code=True)
def _(energy, mo, np, plt, rho, x):
    _active = x < energy.value
    mo.stop(
        not np.any(rho[_active]),
        mo.md("There are no populated accessible centers at this energy."),
    )
    _x = x[_active]
    _dos = rho[_active] / rho[_active].max()
    _root = np.sqrt(energy.value - _x)
    _root /= _root.max()
    _product = _root * _dos
    _fig, _ax = plt.subplots(figsize=(8, 3), layout="constrained")
    for _y, _name in [
        (_dos, "Vibrational DOS"),
        (_root, "Residual square root"),
        (_product / _product.max(), "Product (target)"),
    ]:
        _ax.plot(_x, _y, label=_name)
    _ax.set(xlabel="Vibrational energy (K)", ylabel="Fraction of each curve’s peak")
    _ax.legend()
    mo.vstack(
        [
            mo.md("### Where do the two factors balance?"),
            _fig,
            mo.md(
                "Each curve has its own vertical scale here. On the rising toy DOS, the product "
                "peaks before the last bin because the remaining energy is running out."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What does a good bound look like?

    For each accessible bin choose an envelope height $b_i\ge w_i$. Propose a bin
    with probability $b_i/B$, where $B=\sum_i b_i$. Keep it with probability
    $w_i/b_i$. Their product is $w_i/B$, so conditioning on acceptance gives
    precisely $p_i$. The overall acceptance is $A=Z(E)/B$. **Shaded area between
    the two curves is wasted proposal mass.** Both curves below share one scale.

    | Method | Envelope or draw | When it helps and where it costs |
    |---|---|---|
    | Direct CDF | Draw from $p_i$ directly. | Simple reference. The C++ sampler scans accessible bins to normalize and scans again to draw. A CDF can be reused if $E$ repeats exactly. |
    | DOS only | $b_i=\rho_i\sqrt{E-x_*}$ on accessible bins, where $x_*$ is the first populated center. | One DOS index works for all energies. A steep DOS proposes near $E$, where the square root is smallest. |
    | Group maximum | Within group $G$, use $b_i=\max_{j\in G}\rho_j\sqrt{R_G}$, where $R_G=\max_{j\in G}r_j$. | Uniform draws within a group need only local DOS maxima. Empty bins between narrow spikes waste proposals. |
    | Group DOS | Within $G$, use $b_i=\rho_i\sqrt{R_G}$. | Preserves every spike and gap. Needs local DOS masses and a weighted draw inside the group. |
    | Upper energy | Build a table at $H\ge E$ and use $b_i=w_i(H)$. | Reusable across nearby energies. Can propose inaccessible bins. A wide band on a steep DOS can have very poor acceptance. |
    """)
    return


@app.cell(hide_code=True)
def _(np):
    methods = ["Direct CDF", "DOS only", "Group maximum", "Group DOS", "Upper energy"]

    def groups(x, energy):
        # From the highest accessible center: singleton, then 1, 2, 4, ... bins.
        end = int(np.searchsorted(x, energy, side="left"))
        size = 1
        first = True
        while end:
            start = max(0, end - size)
            yield slice(start, end)
            end = start
            if first:
                first = False
            else:
                size *= 2

    def weights(x, rho, energy):
        return rho * np.sqrt(np.maximum(energy - x, 0))

    def envelopes(x, rho, energy, high):
        # A common DOS scale over the support of the largest proposal cancels.
        used = int(np.searchsorted(x, high))
        scale = np.max(rho[:used], initial=0.0)
        dens = np.zeros_like(rho)
        if scale:
            dens[:used] = rho[:used] / scale
        target = weights(x, dens, energy)
        eligible = (x < energy) & (dens > 0)
        dos = np.zeros_like(dens)
        if eligible.any():
            dos[eligible] = dens[eligible] * np.sqrt(energy - x[eligible][0])
        rectangle, local = np.zeros_like(dens), np.zeros_like(dens)
        for g in groups(x, energy):
            root = np.sqrt(energy - x[g.start])
            rectangle[g] = dens[g].max() * root
            local[g] = dens[g] * root
        return dict(
            zip(
                methods,
                (target, dos, rectangle, local, weights(x, dens, high)),
                strict=True,
            )
        )

    return envelopes, groups, methods


@app.cell(hide_code=True)
def _(methods, mo):
    method = mo.ui.dropdown(methods, value="DOS only", label="Proposal to inspect")
    draws = mo.ui.slider(
        100,
        5000,
        step=100,
        value=1000,
        label="Requested accepted draws",
        show_value=True,
    )
    mo.hstack([method, draws], justify="start")
    return draws, method


@app.cell(hide_code=True)
def _(energy, envelopes, rho, spare, width, x):
    E = energy.value
    H = min(E + spare.value, width * len(x))
    bounds = envelopes(x, rho, E, H)
    target = bounds["Direct CDF"]
    return E, H, bounds, target


@app.cell(hide_code=True)
def _(E, H, bounds, method, mo, np, plt, target, width, x):
    _b = bounds[method.value]
    _fig, (_left, _right) = plt.subplots(1, 2, figsize=(10, 3.4), layout="constrained")
    _used = x < max(H, width)
    _divisor = max(_b.max(), target.max()) or 1
    _left.fill_between(
        x[_used],
        target[_used] / _divisor,
        _b[_used] / _divisor,
        color="tab:orange",
        alpha=0.35,
        label="Rejected mass",
    )
    _left.plot(x[_used], _b[_used] / _divisor, color="tab:orange", label="Envelope")
    _left.plot(x[_used], target[_used] / _divisor, label="Target")
    _left.axvline(E, color="gray", ls=":")
    _left.set(xlabel="Vibrational energy (K)", ylabel="Weight / common peak")
    _left.legend(fontsize=8)
    _accept = [
        float(target.sum() / b.sum()) if target.any() else 0 for b in bounds.values()
    ]
    _right.barh(
        list(bounds),
        _accept,
        color=["tab:blue" if key == method.value else "lightgray" for key in bounds],
    )
    _right.set(xlim=(0, 1.05), xlabel="Acceptance at this E")
    _right.invert_yaxis()
    for _j, _a in enumerate(_accept):
        if _a:
            _right.text(min(_a, 0.85), _j, f" {_a:.1%}", va="center")
    _note = (
        f"The most likely vibrational bin is **{x[np.argmax(target)]:,.1f} K**. "
        f"The selected envelope peaks at **{x[np.argmax(_b)]:,.1f} K**."
        if target.any()
        else "No populated bin is accessible. The simulator returns **0 K**; acceptance is undefined."
    )
    mo.vstack([_fig, mo.md(_note)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Draw some samples

    The toy code below makes all five methods directly comparable. It materializes
    each envelope and samples its CDF with binary search. This exposes the probability
    law and the number of proposals. It does **not** measure the cost of the proposed
    C++ data structures. With the same random seed, compare both the cumulative probability and the density of accepted draws to their exact values.
    The density panel groups neighboring DOS bins for readability. Bar area gives probability;
    the exact curve uses the same display bins. Try **Upper energy** and increase $H-E$ to see what happens to acceptance.
    """)
    return


@app.cell
def _(np):
    def toy_draw(rng, target, bound, count, budget=100_000):
        """Return accepted bin indices and number of proposals actually consumed."""
        cdf = np.cumsum(bound)
        batches = []
        obtained = attempts = 0
        while obtained < count and attempts < budget:
            size = min(count - obtained, budget - attempts)
            u = np.minimum(rng.random(size) * cdf[-1], np.nextafter(cdf[-1], 0))
            idx = cdf.searchsorted(u, side="right")
            keep = rng.random(size) * bound[idx] < target[idx]
            batches.append(idx[keep])
            obtained += int(keep.sum())
            attempts += size
        return np.concatenate(batches), attempts

    return (toy_draw,)


@app.cell(hide_code=True)
def _(bounds, draws, method, mo, np, plt, target, toy_draw, width, x):
    mo.stop(not target.any(), mo.md("No samples to compare at this energy."))
    _samples, _n = toy_draw(
        np.random.default_rng(618), target, bounds[method.value], draws.value
    )
    _fig, (_ax, _pdf) = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    _k = np.flatnonzero(target)[-1] + 1
    _centers = x[:_k]
    _edges = np.append(np.arange(0, _k, max(1, int(np.ceil(_k / 40)))), _k) * width
    _p = target[:_k] / target.sum()
    _exact_mass = np.histogram(_centers, bins=_edges, weights=_p)[0]
    _pdf.stairs(
        _exact_mass / np.diff(_edges),
        _edges,
        color="tab:blue",
        linewidth=2,
        label="Exact density",
        zorder=3,
    )
    _ax.step(
        [0, *_centers, _edges[-1]],
        [0, *np.cumsum(_p), 1],
        where="post",
        lw=2,
        label="Exact bin CDF",
    )
    if _samples.size:
        _observed = np.bincount(_samples, minlength=_k) / len(_samples)
        _ax.step(
            [0, *_centers, _edges[-1]],
            [0, *np.cumsum(_observed), 1],
            where="post",
            label="Accepted draws",
            color="tab:orange",
        )
        _pdf.hist(
            x[_samples],
            bins=_edges,
            density=True,
            alpha=0.45,
            color="tab:orange",
            label="Accepted draws",
        )
    _ax.set(xlabel="Vibrational energy (K)", ylabel="Cumulative probability")
    _pdf.set(xlabel="Vibrational energy (K)", ylabel="Probability density (1/K)")
    for _panel in (_ax, _pdf):
        _panel.legend()
    mo.vstack(
        [
            _fig,
            mo.md(
                f"**{len(_samples):,} accepted out of {_n:,} proposals** "
                f"({len(_samples) / _n:.1%}). The toy stops after 100,000 proposals."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why the local bound works

    **Which samplers use this argument?** Both **Group maximum** and **Group DOS** partition
    the accessible bins and replace $\sqrt{r_i}$ within each group $G$ by its largest
    value $\sqrt{R_G}$. Only **Group DOS** keeps the actual $\rho_i$ in its proposal.
    Its acceptance depends solely on the square root. **Group maximum** also pays a
    factor $\rho_i/\max_{j\in G}\rho_j$, which can be zero. The **tree sampler**
    is another implementation of Group DOS with a different choice of groups.
    Direct CDF, DOS only and Upper energy use the separate constructions in the table above.

    Count backwards from the highest accessible bin. Keep it on its own, then take
    groups of 1, 2, 4, 8, … bins. For example, one group spans bins 4 through 7
    steps back. On an evenly spaced grid the largest residual in a group is at most
    twice its smallest. Even if $E$ is barely above the last center, that center's
    singleton envelope is exact.

    Let $D_G=\sum_{i\in G}\rho_i$. Choosing a group with mass $D_G\sqrt{R_G}$,
    then a bin within it proportional to $\rho_i$, gives exactly the **Group DOS**
    envelope above. The final acceptance test is

    $$a_i=\sqrt{r_i/R_G}\ \ge\ 1/\sqrt{2}.$$

    This gives at most $\sqrt{2}$ proposals on average, regardless of DOS shape.
    Extending the grid above $E$ adds no accessible groups. Refining the grid
    doubles the accessible bins for roughly one extra group.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    example_energy = mo.ui.slider(
        1, 16, step=1, value=13, label="Illustration energy (K)", show_value=True
    )
    mo.vstack(
        [
            mo.md("### Follow the groups across sixteen bins"),
            example_energy,
            mo.md(
                "This small example drives both illustrations. Its bin centers are 0.5, 1.5, …, 15.5 K. "
                "Change the energy by one step and follow the highest accessible bin."
            ),
        ]
    )
    return (example_energy,)


@app.cell
def _(np):
    example_x = np.arange(16) + 0.5
    example_rho = np.array([1, 0, 1, 2, 1, 3, 4, 2, 8, 0, 7, 12, 16, 11, 18, 22])
    return example_rho, example_x


@app.cell(hide_code=True)
def _(example_energy, example_rho, example_x, groups, mo, np, plt):
    _e = example_energy.value
    _fig, (_root, _accept) = plt.subplots(
        2, 1, sharex=True, figsize=(9, 4.5), layout="constrained"
    )
    for _j, _g in enumerate(groups(example_x, _e)):
        _x = example_x[_g]
        _r = np.sqrt(_e - _x)
        _ceiling = _r.max()
        _color = ("#f9e6c3", "#dbeef4")[_j % 2]
        for _ax in (_root, _accept):
            _ax.axvspan(_x[0] - 0.5, _x[-1] + 0.5, color=_color)
        _root.hlines(
            _ceiling,
            _x[0] - 0.5,
            _x[-1] + 0.5,
            color="tab:orange",
            lw=3,
            label="Group ceiling" if _j == 0 else None,
        )
        _root.scatter(
            _x,
            _r,
            color="tab:blue",
            zorder=3,
            label="Residual square root" if _j == 0 else None,
        )
        _present = example_rho[_g] > 0
        _accept.scatter(_x[_present], (_r / _ceiling)[_present], color="tab:blue")
    _accept.axhline(2**-0.5, ls="--", color="black", label="Guaranteed minimum (0.707)")
    for _ax in (_root, _accept):
        _ax.axvline(_e, ls=":", color="gray")
        _ax.set_xlim(0, 16)
        _ax.legend(loc="lower left", fontsize=8)
    _root.set(ylabel="Square root of energy (√K)")
    _accept.set(
        xlabel="Vibrational bin center (K)",
        ylabel="Acceptance probability",
        ylim=(0.56, 1.08),
    )
    mo.vstack(
        [
            _fig,
            mo.md(
                "Each shaded strip is one backward group. Orange is the proposal ceiling; "
                "blue dots in the lower panel are the ratios of the blue heights above to that ceiling. "
                "Empty DOS bins are never proposed by Group DOS."
            ),
        ]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### A tree that avoids subtracting prefixes

    A global prefix CDF can lose a small interval completely. For example, in
    binary64 arithmetic, $(10^{97}+1)-10^{97}=0$ even though the interval mass was
    one. Instead store each node's **local** DOS sum. The toy below uses log masses:
    a leaf stores $\log\rho_i$ (or $-\infty$ for zero); a parent uses
    $\log(\exp a+\exp b)$ computed with `logaddexp`. No large physical-unit sum
    or difference of prefixes is formed.

    Descend until each accessible node satisfies $R_G\le 2\min_{i\in G}r_i$.
    Its subtree already contains the group's DOS mass. These tree-aligned groups
    can differ from the backward groups above but obey the same acceptance bound.
    Construction takes $O(N)$ space and work for $N$ bins, once per DOS. On a
    uniform grid, only a constant number of nodes per level need splitting near
    $E$, and a proposal descends at most $O(\log N)$ levels. For very small $E$,
    a short direct scan could still win.
    """)
    return


@app.cell
def _(np):
    class DosTree:
        def __init__(self, x, rho):
            self.x = x
            self.size = 1 << (len(x) - 1).bit_length()
            self.log_mass = np.full(2 * self.size, -np.inf)
            leaves = self.log_mass[self.size : self.size + len(x)]
            np.log(rho, out=leaves, where=rho > 0)
            level = self.size
            while level > 1:
                children = self.log_mass[level : 2 * level]
                self.log_mass[level // 2 : level] = np.logaddexp(
                    children[::2], children[1::2]
                )
                level //= 2

        def cover(self, energy):
            nodes = []

            def visit(node, left, right):
                if (
                    left >= len(self.x)
                    or self.x[left] >= energy
                    or not np.isfinite(self.log_mass[node])
                ):
                    return
                largest = energy - self.x[left]
                if right <= len(self.x) and largest <= 2 * (energy - self.x[right - 1]):
                    nodes.append((node, largest))
                    return
                middle = (left + right) // 2
                visit(2 * node, left, middle)
                visit(2 * node + 1, middle, right)

            visit(1, 0, self.size)
            return nodes

        def draw(self, rng, energy, count):
            cover = self.cover(energy)
            if not cover:
                return np.zeros(count), 0, 0
            nodes, residuals = map(np.asarray, zip(*cover, strict=True))
            logs = self.log_mass[nodes] + 0.5 * np.log(residuals)
            cdf = np.cumsum(np.exp(logs - logs.max()))
            samples = []
            attempts = 0
            while len(samples) < count:
                attempts += 1
                u = min(rng.random() * cdf[-1], np.nextafter(cdf[-1], 0))
                j = cdf.searchsorted(u, side="right")
                node = nodes[j]
                while node < self.size:
                    child = 2 * node
                    prob_left = np.exp(self.log_mass[child] - self.log_mass[node])
                    node = child if rng.random() < prob_left else child + 1
                value = self.x[node - self.size]
                if rng.random() ** 2 * residuals[j] < energy - value:
                    samples.append(value)
            return np.asarray(samples), attempts, len(cover)

    return (DosTree,)


@app.cell(hide_code=True)
def _(DosTree, example_energy, example_rho, example_x, mo, np, plt):
    _tree = DosTree(example_x, example_rho)
    _e = example_energy.value
    _cover = dict(_tree.cover(_e))
    _fig, _ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
    _colors = {"group": "#bce5d6", "split": "#ffebc4", "skip": "#eeeeee"}

    def _draw(node, lo, hi, depth, parent=None):
        mid = (lo + hi) / 2
        if parent is not None:
            _ax.plot([parent, mid], [1 - depth, -depth], color="#758390", zorder=0)
        mass = int(example_rho[lo:hi].sum())
        kind = (
            "group"
            if node in _cover
            else "skip"
            if example_x[lo] >= _e or mass == 0
            else "split"
        )
        if kind == "split":
            cut = (lo + hi) // 2
            _draw(node * 2, lo, cut, depth + 1, mid)
            _draw(node * 2 + 1, cut, hi, depth + 1, mid)
        interval = f"{lo}–{hi - 1}" if hi - lo > 1 else str(lo)
        _ax.text(
            mid,
            -depth,
            f"{interval}\nD={mass}",
            ha="center",
            va="center",
            fontsize=8,
            bbox=dict(
                boxstyle="round,pad=0.35", facecolor=_colors[kind], edgecolor="#53616c"
            ),
        )

    _draw(1, 0, len(example_x), 0)
    _ax.set(
        xlim=(-0.3, 16.3),
        ylim=(-4.6, 0.6),
        title=f"Tree cover at E = {_e} K   •   split (amber)   /   use group (green)   /   skip (gray)",
    )
    _ax.axis("off")
    _labels = [
        f"{int(round(np.exp(_tree.log_mass[n])))} × √{r:g}" for n, r in _cover.items()
    ]
    mo.vstack(
        [
            _fig,
            mo.md(
                "Each box names its **inclusive bin indices** and its **DOS sum D**. "
                "Read down from the root. Green boxes are the groups actually returned by "
                "`DosTree.cover`; gray branches are inaccessible or empty. To propose a bin, "
                "first choose a green box proportional to its DOS sum times its residual ceiling, "
                "then descend within that box proportional to child DOS mass."
            ),
            mo.md(
                "For this energy, the green group weights from left to right are **"
                + ", ".join(_labels)
                + "**."
            ),
        ]
    )
    return


@app.cell
def _(DosTree, rho, x):
    tree = DosTree(x, rho)
    return (tree,)


@app.cell(hide_code=True)
def _(E, draws, mo, np, target, tree, x):
    _values, _attempts, _groups = tree.draw(np.random.default_rng(619), E, draws.value)
    if _attempts:
        _exact = np.average(x, weights=target)
        _message = (
            f"At this E the tree uses **{_groups} groups** and accepted "
            f"**{len(_values) / _attempts:.1%}** of {_attempts:,} proposals. "
            f"Mean vibrational energy: **{_values.mean():,.1f} K** sampled, "
            f"**{_exact:,.1f} K** exact."
        )
    else:
        _message = "Empty cover. The tree returned zero energy for every draw."
    mo.md(_message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## When would a reusable upper-energy table win?

    Fix a band $[L,H]$ and prepare a sampler for $w_i(H)$ once. Since $w_i$ increases
    with available energy for every bin, the table covers every $E$ in the band.
    An inaccessible proposal is rejected; otherwise the acceptance test is
    $\sqrt{(E-x_i)/(H-x_i)}$. Overall acceptance is $Z(E)/Z(H)$, with its minimum at $L$.

    **Choose bands by growth of $Z$.** For example, if $Z(L)/Z(H)\ge 1/2$, a cached
    table costs at most two proposals on average throughout that band. An alias table
    gives constant-time proposals after linear setup. In the plot, $H$ stays fixed
    while $E$ moves along the horizontal axis. Where the curve falls below your desired
    acceptance, narrow the band or use the local tree. A cache is attractive when many
    collisions revisit a band; multiple large tables are expensive for one-off energies.
    """)
    return


@app.cell(hide_code=True)
def _(E, H, mo, np, plt, rho, x):
    _cut = int(np.searchsorted(x, H))
    _scale = np.max(rho[:_cut], initial=0.0)
    mo.stop(not _scale, mo.md("Raise the upper energy to cover a populated bin."))
    _dens = rho[:_cut] / _scale

    def _z(e):
        return np.dot(_dens, np.sqrt(np.maximum(e - x[:_cut], 0)))

    _es = np.linspace(0, H, 121)
    _rates = np.array([_z(e) for e in _es]) / _z(H)
    _fig, _ax = plt.subplots(figsize=(8, 3), layout="constrained")
    _ax.plot(_es, _rates)
    _ax.scatter([E], [_z(E) / _z(H)], label="Selected E")
    _ax.axhline(0.5, color="gray", linestyle="--", label="50% acceptance")
    _ax.set(
        xlabel="Actual available energy E (K)",
        ylabel="Upper-table acceptance",
        title=f"One cached table at H = {H:,} K",
        ylim=(0, 1.05),
    )
    _ax.legend()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What I would benchmark in apitofsim

    Start with the local DOS tree and a short direct scan. The tree pays its setup
    cost once per cluster, even when collisions have different energies. Use the
    resolution control on **Rising DOS** to separate the cost of finer bins from
    changes in acceptance. Try **1A_neutral** with **Group maximum** to see the
    cost of proposing empty bins. Try **Upper energy** with zero headroom to see
    perfect acceptance and consider how often that exact table could be reused.

    Before choosing a C++ default, replay energies from both collision and rotational
    draws. Measure setup time, memory per cluster, and time per draw separately.
    Compare output bin probabilities to direct enumeration, including the first
    populated bin and the final half-bin below the histogram edge. The Python toy
    proposal counts answer the distribution question; those native timings would
    answer the speed question.
    """)
    return


if __name__ == "__main__":
    app.run()
