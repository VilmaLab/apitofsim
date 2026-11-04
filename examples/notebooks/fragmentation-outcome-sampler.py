# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo>=0.23.9,<1",
#     "numpy>=2,<3",
#     "scipy>=1.14,<2",
#     "matplotlib>=3.9,<4",
#     "duckdb>=1.4,<2",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import csv
    import io
    from html import escape
    from pathlib import Path

    import duckdb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.special import logsumexp

    return Path, csv, duckdb, escape, io, logsumexp, mo, np, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # After a cluster breaks: sampling the charged daughter

    A serial fragmentation simulation needs a velocity, angular velocity and
    vibrational energy for the surviving ion. Setting all three to zero loses the
    parent's centre-of-mass motion and the energy remaining after bond breaking.
    Here we derive and explore a statistical replacement, using real cluster data.

    **Start here:** load a prepared apitofsim DuckDB, choose a channel and DOS grid,
    then submit a parent state. The figures compare sampled outcomes with the
    target distribution, and an SVG follows one selected breakup. The derivation
    remains readable without a database. No native apitofsim build is required.

    The starting point is [Zapadinsky et al. (2019), *Modeling on Fragmentation of
    Clusters inside a Mass Spectrometer*, Eqs. 23–39](https://doi.org/10.1021/acs.jpca.8b10744).
    The paper derives the **zero-angular-momentum** kernel. Its extension here to
    a specified parent angular momentum is a proposed model, derived below. We
    retain spherical rigid rotors, statistical vibrational DOS and the paper's
    $b_{\max}=\infty$ approximation. This is a distribution **conditional on the
    chosen fragmentation channel**, not a calculation of branching probabilities
    or a replacement for the simulator's rate constants.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.Html("""
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 860 270" role="img"
      aria-label="A moving rotating parent splits into a charged daughter and a neutral. Both inherit its centre-of-mass velocity and receive opposite recoil momenta.">
      <defs><marker id="overview-arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0L8 4L0 8" fill="#475569"/></marker></defs>
      <rect width="860" height="270" rx="12" fill="#f8fafc"/>
      <g font-family="sans-serif" font-size="16" fill="#172554">
      <circle cx="110" cy="125" r="48" fill="#c7d2fe" stroke="#4338ca" stroke-width="2"/>
      <text x="110" y="131" text-anchor="middle">P±</text>
      <path d="M67 77 Q110 36 153 77" fill="none" stroke="#475569" stroke-width="2" marker-end="url(#overview-arrow)"/>
      <text x="110" y="32" text-anchor="middle">parent spin J</text>
      <path d="M167 125H292" stroke="#475569" stroke-width="3" marker-end="url(#overview-arrow)"/>
      <text x="223" y="106" text-anchor="middle">velocity V</text>
      <text x="317" y="131">→</text>
      <circle cx="480" cy="83" r="34" fill="#99f6e4" stroke="#0f766e" stroke-width="2"/>
      <circle cx="480" cy="192" r="23" fill="#fed7aa" stroke="#c2410c" stroke-width="2"/>
      <text x="480" y="89" text-anchor="middle">D±</text>
      <text x="480" y="198" text-anchor="middle">N</text>
      <path d="M520 83H675" stroke="#475569" stroke-width="2" marker-end="url(#overview-arrow)"/>
      <path d="M510 192H625" stroke="#475569" stroke-width="2" marker-end="url(#overview-arrow)"/>
      <text x="699" y="88">V + recoil</text><text x="648" y="197">V − recoil</text>
      <text x="365" y="34">spin j_D</text><text x="365" y="238">spin j_N</text>
      <text x="110" y="221" text-anchor="middle">E = vibration + rotation</text>
      <text x="606" y="250" text-anchor="middle">J = j_D + j_N + orbital angular momentum ℓ</text>
      </g>
    </svg>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Symbols and energy accounting

    A parent $P$ breaks into a charged daughter $D$ and neutral $N$.
    All vectors use the laboratory axes. $\mathbf J$ and $\boldsymbol\ell$ refer to
    the parent centre of mass; each product spin is about that product's own centre
    of mass. Excitation energies are above each species' ground state.

    | Symbol | Meaning |
    |---|---|
    | $m_i,I_i$ | Mass and scalar moment of inertia of species $i=P,D,N$ |
    | $\mathbf V,\boldsymbol\omega_P$ | Parent velocity and angular velocity immediately before breakup |
    | $\mathbf J=I_P\boldsymbol\omega_P$ | Parent angular momentum in the spherical model |
    | $E_{v,i}$, $E_{r,i}$ | Vibrational and rotational excitation energies |
    | $E=E_{v,P}+E_{r,P}$ | Parent internal energy; excludes laboratory translation |
    | $D_0$ | Channel fragmentation energy, with consistent ground-state energy references |
    | $Q=E-D_0$ | Energy available to separated products |
    | $K=\mu|\mathbf v_D-\mathbf v_N|^2/2$ | Relative translational (recoil) energy |
    | $V=E_{v,D}+E_{v,N}$ | Total product vibrational energy (scalar $V$, not vector $\mathbf V$) |
    | $\mathbf n$ | Unit vector along the outgoing relative velocity |
    | $\boldsymbol\ell$, $b$ | Orbital angular momentum and asymptotic impact parameter |
    | $\mathbf j_D,\mathbf j_N$ | Product spin angular momenta |
    | $\rho_i(u)$ | Vibrational DOS of product $i$ at energy $u$ |

    Define
    $$M=m_D+m_N,\quad \mu=\frac{m_Dm_N}{M},\quad
    I_\Sigma=I_D+I_N,\quad I_*=\frac{I_DI_N}{I_\Sigma}.$$
    The constraints are
    $$\mathbf j_D+\mathbf j_N+\boldsymbol\ell=\mathbf J,\qquad
    \boldsymbol\ell\cdot\mathbf n=0,\qquad |\boldsymbol\ell|=b\sqrt{2\mu K},$$
    $$K+\frac{|\mathbf j_D|^2}{2I_D}
       +\frac{|\mathbf j_N|^2}{2I_N}+V=Q.$$
    The combined vibrational DOS is the convolution
    $$\rho_{DN}(V)=\int_0^V\rho_D(u)\rho_N(V-u)\,du.$$

    **Units:** equations use physical energies (J). Controls and energy plots use
    energy divided by $k_B$, in K; these numbers are not temperatures of an
    equilibrated cluster. Velocities use m/s, angular velocities rad/s, masses kg,
    inertias kg m² and angular momenta kg m²/s. The code converts explicitly at
    reconstruction. It uses the same constants and mass convention as apitofsim.
    For stored principal rotational temperatures $\Theta_1,\Theta_2,\Theta_3$,
    its effective spherical inertia is
    $I=\hbar^2/[2k_B(\Theta_1\Theta_2\Theta_3)^{1/3}]$, where $\hbar$ is the
    reduced Planck constant and $k_B$ is Boltzmann's constant.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## From the paper's rate integral to a sampling distribution

    Detailed balance weights product states by collision **flux**. The translational
    DOS contributes $\sqrt K$ and the relative speed contributes another $\sqrt K$.
    With cross-section element $b\,db\,d\phi$, the paper's Eq. 35 is proportional to
    $$K\,b\,dK\,db\,d\phi\,d\Omega_{\mathbf n}\,d^3j_D\,
    \rho_{DN}\!\left(Q-K-\frac{|\mathbf j_D|^2}{2I_D}
    -\frac{|\mathbf J-\boldsymbol\ell-\mathbf j_D|^2}{2I_N}\right).$$
    Writing $\mathbf J$ here extends the paper's $\mathbf J=0$ constraint.
    Here $d\Omega$ denotes a solid-angle element and $\phi$ is the azimuth of
    the impact parameter about $\mathbf n$. The density is zero whenever its
    argument is negative.

    Complete the square with
    $$\mathbf s=\mathbf j_D-\frac{I_D}{I_\Sigma}(\mathbf J-\boldsymbol\ell),$$
    $$E_{r,D}+E_{r,N}=\frac{|\mathbf s|^2}{2I_*}
                     +\frac{|\mathbf J-\boldsymbol\ell|^2}{2I_\Sigma}.$$
    For a chosen $\mathbf n$, let
    $$J_\parallel=\mathbf J\cdot\mathbf n,\qquad
    \mathbf J_\perp=\mathbf J-J_\parallel\mathbf n,$$
    $$R=\frac{|\mathbf s|^2}{2I_*},\quad
    A=\frac{|\boldsymbol\ell-\mathbf J_\perp|^2}{2I_\Sigma},\quad
    Q_{\mathbf n}=Q-\frac{J_\parallel^2}{2I_\Sigma}.$$
    Thus $K+A+R+V=Q_{\mathbf n}$. The auxiliary energies $A,R$ are not the
    individual products' rotational energies.

    The Jacobians are $K b\,db\,d\phi=d^2\ell/(2\mu)$,
    $d^2\ell=I_\Sigma\,dA\,d\phi$ and
    $d^3s\propto\sqrt R\,dR\,d\Omega_{\mathbf s}$. The joint measure simplifies to
    $$dW\propto\sqrt R\,\rho_{DN}(Q_{\mathbf n}-K-A-R)
    \,dK\,dA\,dR\,d\Omega_{\mathbf n}\,d\phi\,d\Omega_{\mathbf s}.$$
    At fixed $V$, integrating the mechanical split gives
    $$\int\sqrt R\,\delta(W-K-A-R)\,dK\,dA\,dR=\frac4{15}W^{5/2},
    \qquad W=Q_{\mathbf n}-V.$$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The sampler

    Define $Z(x)=\int_0^x\rho_{DN}(V)(x-V)^{5/2}\,dV$ for $x>0$, otherwise zero.
    The recoil direction has density $p(\mathbf n)\propto Z(Q_{\mathbf n})$ per unit
    solid angle. Conditional on that direction,
    $$p(V\mid\mathbf n)=\frac{\rho_{DN}(V)(Q_{\mathbf n}-V)^{5/2}}{Z(Q_{\mathbf n})},$$
    $$\frac{(K,A,R)}{Q_{\mathbf n}-V}\sim\operatorname{Dirichlet}(1,1,3/2).$$
    A Dirichlet draw is obtained by normalizing independent unit-scale gamma draws
    with the listed shape parameters. Draw $\mathbf e$ uniformly on the unit circle
    perpendicular to $\mathbf n$ and $\mathbf a$ uniformly on the unit sphere. Set
    $$\boldsymbol\ell=\mathbf J_\perp+\sqrt{2I_\Sigma A}\,\mathbf e,
    \qquad \mathbf s=\sqrt{2I_*R}\,\mathbf a,$$
    $$\mathbf j_D=\frac{I_D}{I_\Sigma}(\mathbf J-\boldsymbol\ell)+\mathbf s,
    \qquad \mathbf j_N=\frac{I_N}{I_\Sigma}(\mathbf J-\boldsymbol\ell)-\mathbf s.$$
    Split the vibrational energy using
    $$p(E_{v,D}=u\mid V)=\frac{\rho_D(u)\rho_N(V-u)}{\rho_{DN}(V)}.$$
    Finally assign
    $$\boxed{\mathbf v_D=\mathbf V+\frac{m_N}{M}\sqrt{\frac{2K}{\mu}}\mathbf n,
    \quad\boldsymbol\omega_D=\frac{\mathbf j_D}{I_D},\quad E_{v,D}=u.}$$
    The neutral receives $\mathbf v_N=\mathbf V-(m_D/M)\sqrt{2K/\mu}\mathbf n$.
    These are asymptotic product velocities, applied instantaneously at the breakup
    location on the much larger instrument scale.

    **Check against the paper:** at $J=0$, $\widetilde K=K+A=(1+\mu b^2/I_\Sigma)K$
    and $\widetilde R=R$. Integrating out $K\in[0,\widetilde K]$ gives
    $$p(\widetilde K,\widetilde R)\propto\widetilde K\sqrt{\widetilde R}
    \rho_{DN}(Q-\widetilde K-\widetilde R),$$
    exactly the kernel of Eq. 39. The physical recoil is
    $K\mid\widetilde K\sim\mathrm{Uniform}(0,\widetilde K)$, not $\widetilde K$ itself.

    **Atom-like neutral:** it has no spin or vibrations. Use $I_\Sigma=I_D$, omit
    $R,\mathbf s$, change the exponent $5/2$ to $1$, draw $(K,A)/W$ from
    $\mathrm{Dirichlet}(1,1)$, and set $\mathbf j_D=\mathbf J-\boldsymbol\ell$.
    The notebook handles that case without dividing by a zero neutral inertia.
    """)
    return


@app.cell(hide_code=True)
def _(Path, mo):
    _default = Path(__file__).resolve().parents[1] / "besel2020/database.duckdb"
    database_path = mo.ui.text(
        value=str(_default) if _default.is_file() else "",
        placeholder="/path/to/prepared/database.duckdb",
        label="Prepared database (read-only)",
        full_width=True,
    )
    mo.vstack([mo.md("## Load a database"), database_path])
    return (database_path,)


@app.cell
def _(duckdb, np):
    # Match src/include/consts.h and compute_derived's spherical approximation.
    KB = 1.38064852e-23
    HBAR = 1.054571800e-34
    PROTON_MASS = 1.6726219e-27
    HARTREE_K = 3.157732e5

    def load_catalog(path):
        with duckdb.connect(str(path), read_only=True) as db:
            channels = db.execute("""
                select p.id, c.common_name, d.common_name, n.common_name
                from pathway p join cluster c on c.id=p.cluster_id
                join cluster d on d.id=p.product1_id
                join cluster n on n.id=p.product2_id
                where c.charge != 0 and
                  ((d.charge=c.charge and n.charge=0) or
                   (n.charge=c.charge and d.charge=0))
                order by p.id
            """).fetchall()
            grids = db.execute("""
                select h.id, h.bin_width, h.max from histogram_params h
                where exists (select 1 from cluster_dos d
                              where d.histogram_params_id=h.id)
                order by h.bin_width, h.id
            """).fetchall()
        return channels, grids

    def load_channel(path, channel_id, grid_id):
        with duckdb.connect(str(path), read_only=True) as db:
            ids = db.execute(
                "select cluster_id, product1_id, product2_id from pathway where id=?",
                [channel_id],
            ).fetchone()
            width = db.execute(
                "select bin_width from histogram_params where id=?", [grid_id]
            ).fetchone()[0]
            species = []
            for cid in ids:
                row = db.execute(
                    """
                    select common_name, atomic_mass, charge, electronic_energy,
                           rotational_temperatures, vibrational_temperatures
                    from cluster where id=?
                """,
                    [cid],
                ).fetchone()
                name, mass, charge, energy, rotations, vibrations = row
                rotations = np.asarray(rotations, dtype=float)
                atom = (not vibrations) and not np.any(
                    np.isfinite(rotations) & (rotations > 0)
                )
                if not atom and (
                    rotations.shape != (3,)
                    or not np.all(np.isfinite(rotations) & (rotations > 0))
                ):
                    raise ValueError(
                        f"{name}: need three positive rotational temperatures; linear rotors are not supported."
                    )
                inertia = (
                    0.0
                    if atom
                    else HBAR**2 / (2 * KB * np.exp(np.log(rotations).mean()))
                )
                rows = db.execute(
                    """
                    select data from cluster_dos
                    where cluster_id=? and histogram_params_id=?
                """,
                    [cid, grid_id],
                ).fetchall()
                if len(rows) > 1:
                    raise ValueError(f"{name}: duplicate DOS tables on grid {grid_id}.")
                if not atom and not rows:
                    raise ValueError(
                        f"{name}: missing DOS on grid {grid_id}; choose another prepared grid."
                    )
                species.append(
                    dict(
                        id=cid,
                        name=name or str(cid),
                        mass=float(mass) * PROTON_MASS,
                        inertia=inertia,
                        charge=charge,
                        electronic_energy=energy,
                        atom=atom,
                        rho=None if atom else np.asarray(rows[0][0], dtype=float),
                    )
                )
            parent, first, second = species
            daughter, neutral = (first, second) if first["charge"] else (second, first)
            if parent["atom"] or daughter["atom"]:
                raise ValueError(
                    "This notebook requires a molecular parent and charged daughter."
                )
            total_mass = daughter["mass"] + neutral["mass"]
            if not np.isclose(parent["mass"], total_mass, rtol=1e-6, atol=0):
                raise ValueError(
                    "Parent and product masses do not balance; check the channel data."
                )
            if not all(np.isfinite(s["mass"]) and s["mass"] > 0 for s in species):
                raise ValueError("All masses must be finite and positive.")
            threshold = (
                daughter["electronic_energy"]
                + neutral["electronic_energy"]
                - parent["electronic_energy"]
            ) * HARTREE_K
            if not np.isfinite(threshold):
                raise ValueError("The channel has no finite fragmentation energy.")
            rates = (
                db.execute(
                    """
                select h.id, h.bin_width, k.data from k_rate k
                join histogram_params h on h.id=k.histogram_params_id
                where k.pathway_id=? order by h.bin_width, h.id
            """,
                    [channel_id],
                ).fetchall()
                if db.execute(
                    "select count(*) from information_schema.tables where table_name='k_rate'"
                ).fetchone()[0]
                else []
            )
        return dict(
            parent=parent,
            daughter=daughter,
            neutral=neutral,
            width=float(width),
            threshold=threshold,
            rates=rates,
            channel_id=channel_id,
            grid_id=grid_id,
            path=str(path),
        )

    return KB, load_catalog, load_channel


@app.cell(hide_code=True)
def _(Path, database_path, duckdb, load_catalog, mo):
    mo.stop(
        not database_path.value.strip(),
        mo.md("Enter a prepared database path to enable the experiment below."),
    )
    db_path = Path(database_path.value).expanduser()
    mo.stop(not db_path.is_file(), mo.md("That database file does not exist."))
    try:
        channel_rows, grid_rows = load_catalog(db_path)
    except (duckdb.Error, OSError) as _error:
        mo.stop(
            True,
            mo.callout(f"Cannot read this prepared database: {_error}", kind="warn"),
        )
    mo.stop(
        not channel_rows or not grid_rows,
        mo.md(
            "Prepare charged fragmentation channels and vibrational DOS tables first."
        ),
    )
    channel_menu = mo.ui.dropdown(
        {f"{pid}: {p} → {a} + {b}": pid for pid, p, a, b in channel_rows},
        value=f"{channel_rows[0][0]}: {channel_rows[0][1]} → {channel_rows[0][2]} + {channel_rows[0][3]}",
        label="Fragmentation channel",
        searchable=True,
        full_width=True,
    )
    grid_menu = mo.ui.dropdown(
        {
            f"{gid}: Δ = {width:g} K, maximum = {high:g} K": gid
            for gid, width, high in grid_rows
        },
        value=f"{grid_rows[0][0]}: Δ = {grid_rows[0][1]:g} K, maximum = {grid_rows[0][2]:g} K",
        label="Prepared DOS grid",
        full_width=True,
    )
    mo.vstack([channel_menu, grid_menu])
    return channel_menu, db_path, grid_menu


@app.cell(hide_code=True)
def _(channel_menu, db_path, duckdb, grid_menu, load_channel, mo):
    try:
        channel = load_channel(db_path, channel_menu.value, grid_menu.value)
    except (duckdb.Error, ValueError, TypeError, OSError) as _error:
        mo.stop(True, mo.callout(str(_error), kind="warn"))
    mo.vstack(
        [
            mo.md(
                f"**Channel {channel['channel_id']}:** {channel['parent']['name']} → "
                f"{channel['daughter']['name']} (charged) + {channel['neutral']['name']} (neutral). "
                f"Database threshold: **{channel['threshold']:.2f} K**."
            ),
            mo.ui.table(
                [
                    {
                        "role": role,
                        "cluster": channel[role]["name"],
                        "mass (kg)": channel[role]["mass"],
                        "inertia (kg m²)": channel[role]["inertia"],
                        "atom-like": channel[role]["atom"],
                    }
                    for role in ["parent", "daughter", "neutral"]
                ],
                selection=None,
                show_column_summaries=False,
            ),
            mo.md(
                "The threshold is the stored product-minus-parent electronic energy difference "
                "converted from hartree. Confirm the inputs include the intended zero-point corrections; "
                "use the override below if your simulation used another threshold."
            ),
        ]
    )
    return (channel,)


@app.cell(hide_code=True)
def _(channel, mo):
    default_parameters = dict(
        excess_K=3000.0,
        rotation_K=300.0,
        speed=1000.0,
        velocity_angle=0.0,
        count=5000,
        seed=42,
        max_bins=2000,
        threshold_K=float(channel["threshold"]),
    )
    controls = mo.ui.dictionary(
        {
            "excess_K": mo.ui.number(
                start=0.01, value=3000, label="Excess energy Q/kB (K)"
            ),
            "rotation_K": mo.ui.number(
                start=0,
                value=300,
                label="Parent rotation Er,P/kB (K); set 0 for the paper's limit",
            ),
            "speed": mo.ui.number(start=0, value=1000, label="Parent speed |V| (m/s)"),
            "velocity_angle": mo.ui.slider(
                0,
                180,
                step=1,
                value=0,
                label="Angle between V and +z (degrees)",
                show_value=True,
            ),
            "count": mo.ui.slider(
                100,
                20000,
                step=100,
                value=5000,
                label="Number of breakups",
                show_value=True,
            ),
            "seed": mo.ui.number(
                start=0,
                stop=2**32 - 1,
                value=42,
                step=1,
                label="Random seed; change to resample",
            ),
            "max_bins": mo.ui.dropdown(
                {"500": 500, "1,000": 1000, "2,000": 2000, "4,000": 4000},
                value="2,000",
                label="Maximum accessible DOS bins per product",
            ),
            "threshold_K": mo.ui.number(
                value=float(channel["threshold"]),
                label="Fragmentation threshold D0/kB (K)",
            ),
        }
    ).form(submit_button_label="Sample fragmentation outcomes")
    mo.vstack(
        [
            mo.md("## Choose the parent state"),
            mo.md(
                "Choose axes so that **J points along +z**. The velocity angle then sets its "
                "orientation relative to the spin. Defaults run immediately; submit to apply edits."
            ),
            controls,
        ]
    )
    return controls, default_parameters


@app.cell
def _(KB, logsumexp, np):
    def probabilities(log_weights):
        total = logsumexp(log_weights)
        if not np.isfinite(total):
            raise ValueError(
                "No populated accessible product states. Increase Q or use a finer DOS grid."
            )
        return np.exp(log_weights - total)

    def prepare_model(channel, parameters):
        q = float(parameters["excess_K"])
        rot = float(parameters["rotation_K"])
        threshold = float(parameters["threshold_K"])
        if not np.all(
            np.isfinite(
                [q, rot, threshold, parameters["speed"], parameters["velocity_angle"]]
            )
        ):
            raise ValueError("Parent-state inputs must be finite.")
        if q <= 0 or rot < 0 or threshold + q - rot < 0:
            raise ValueError(
                "Require Q > 0, rotation ≥ 0 and parent vibration D0 + Q − rotation ≥ 0."
            )
        base = channel["width"]
        if not np.isfinite(base) or base <= 0:
            raise ValueError("The stored DOS bin width must be finite and positive.")
        factor = max(1, int(np.ceil(q / (base * int(parameters["max_bins"])))))
        width = base * factor

        def product_table(species):
            rho = species["rho"]
            groups = min(int(np.ceil(q / width)), len(rho) // factor)
            if groups * width < q:
                raise ValueError(
                    f"{species['name']}: Q exceeds the complete available DOS bins. Lower Q or increase the bin budget."
                )
            accessible = rho[: groups * factor]
            if not np.all(np.isfinite(accessible) & (accessible >= 0)):
                raise ValueError(
                    f"{species['name']}: nonfinite or negative accessible DOS values."
                )
            logs = np.full(accessible.shape, -np.inf)
            positive = accessible > 0
            logs[positive] = np.log(accessible[positive]) + np.log(base)
            return logsumexp(logs.reshape(-1, factor), axis=1)

        ld = product_table(channel["daughter"])
        atom = channel["neutral"]["atom"]
        ln = np.array([0.0]) if atom else product_table(channel["neutral"])
        if atom:
            vibration = width * (np.arange(len(ld)) + 0.5)
            combined = ld.copy()
        else:
            # Direct log convolution avoids cancellation/negative tails from an FFT.
            size = min(len(ld) + len(ln) - 1, int(np.ceil(q / width)))
            combined = np.empty(size)
            for k in range(size):
                i = np.arange(max(0, k - len(ln) + 1), min(k + 1, len(ld)))
                combined[k] = logsumexp(ld[i] + ln[k - i])
            vibration = width * (np.arange(size) + 1)
        eligible = (vibration < q) & np.isfinite(combined)
        indices = np.flatnonzero(eligible)
        v = vibration[eligible]
        log_rho = combined[eligible]
        inertia_sum = channel["daughter"]["inertia"] + channel["neutral"]["inertia"]
        c_energy = channel["parent"]["inertia"] / inertia_sum * rot
        power = 1.0 if atom else 2.5
        w = q - v
        cmax = (
            np.ones_like(w) if c_energy == 0 else np.minimum(1, np.sqrt(w / c_energy))
        )
        proposal_logs = log_rho + power * np.log(w) + np.log(cmax)
        proposal = probabilities(proposal_logs)
        # Independent Gauss–Legendre integration for plotted marginal targets.
        nodes, weights = np.polynomial.legendre.leggauss(64)
        alpha = c_energy * cmax**2 / w
        mean_accept = (np.maximum(1 - alpha[:, None] * nodes**2, 0) ** power) @ (
            weights / 2
        )
        marginal_logs = proposal_logs + np.log(mean_accept)
        return dict(
            q=q,
            rot=rot,
            threshold=threshold,
            width=width,
            factor=factor,
            ld=ld,
            ln=ln,
            atom=atom,
            indices=indices,
            v=v,
            log_rho=log_rho,
            cmax=cmax,
            proposal=proposal,
            power=power,
            c_energy=c_energy,
            inertia_sum=inertia_sum,
            target_v=probabilities(marginal_logs),
            angular_log_norm=logsumexp(marginal_logs) + np.log(2),
            expected_acceptance=float(proposal @ mean_accept),
        )

    def angular_density(model, cos_theta):
        remaining = (
            model["q"]
            - model["v"][None, :]
            - model["c_energy"] * np.asarray(cos_theta)[:, None] ** 2
        )
        logs = np.full(remaining.shape, -np.inf)
        positive = remaining > 0
        logs[positive] = model["power"] * np.log(remaining[positive])
        return np.exp(
            logsumexp(logs + model["log_rho"], axis=1) - model["angular_log_norm"]
        )

    def sample_outcomes(channel, model, parameters):
        rng = np.random.default_rng(int(parameters["seed"]))
        count = int(parameters["count"])
        if not 1 <= count <= 20000:
            raise ValueError("Choose between 1 and 20,000 samples.")
        chosen, cosines = [], []
        accepted = proposed = collected = 0
        while collected < count:
            batch = max(128, 2 * (count - collected))
            slots = rng.choice(len(model["v"]), size=batch, p=model["proposal"])
            c = rng.uniform(-1, 1, batch) * model["cmax"][slots]
            w = model["q"] - model["v"][slots]
            keep = (
                rng.random(batch)
                < np.maximum(1 - model["c_energy"] * c**2 / w, 0) ** model["power"]
            )
            chosen.append(slots[keep])
            cosines.append(c[keep])
            collected += int(keep.sum())
            accepted += int(keep.sum())
            proposed += batch
            if proposed > 100 * count + 10000:
                raise ValueError(
                    "Unexpectedly low acceptance; inspect the input DOS and energies."
                )
        slots = np.concatenate(chosen)[:count]
        c = np.concatenate(cosines)[:count]
        total_v = model["v"][slots]
        phi = rng.uniform(0, 2 * np.pi, count)
        sin_theta = np.sqrt(np.maximum(1 - c**2, 0))
        n = np.column_stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), c])
        remaining = model["q"] - total_v - model["c_energy"] * c**2
        if model["atom"]:
            fractions = rng.dirichlet([1, 1], count)
            k, a = (remaining[:, None] * fractions).T
            r = np.zeros(count)
        else:
            fractions = rng.dirichlet([1, 1, 1.5], count)
            k, a, r = (remaining[:, None] * fractions).T
        # Two orthonormal tangents avoid a near-parallel Gaussian projection.
        tangent = np.column_stack([c * np.cos(phi), c * np.sin(phi), -sin_theta])
        azimuth = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros(count)])
        psi = rng.uniform(0, 2 * np.pi, count)
        e = np.cos(psi)[:, None] * tangent + np.sin(psi)[:, None] * azimuth
        ip = channel["parent"]["inertia"]
        id_ = channel["daughter"]["inertia"]
        in_ = channel["neutral"]["inertia"]
        isum = id_ + in_
        j_parent = np.array([0.0, 0.0, np.sqrt(2 * ip * KB * model["rot"])])
        j_perp = j_parent - (n @ j_parent)[:, None] * n
        ell = j_perp + np.sqrt(2 * isum * KB * a)[:, None] * e
        if model["atom"]:
            jd = j_parent - ell
            jn = np.zeros_like(jd)
        else:
            spin_direction = rng.normal(size=(count, 3))
            spin_direction /= np.linalg.norm(spin_direction, axis=1)[:, None]
            s = np.sqrt(2 * (id_ * in_ / isum) * KB * r)[:, None] * spin_direction
            jd = id_ / isum * (j_parent - ell) + s
            jn = in_ / isum * (j_parent - ell) - s
        vib_d = np.empty(count)
        if model["atom"]:
            vib_d[:] = total_v
        else:
            combined_indices = model["indices"][slots]
            for index in np.unique(combined_indices):
                matches = np.flatnonzero(combined_indices == index)
                i = np.arange(
                    max(0, index - len(model["ln"]) + 1),
                    min(index + 1, len(model["ld"])),
                )
                split = probabilities(model["ld"][i] + model["ln"][index - i])
                picked = rng.choice(i, size=len(matches), p=split)
                vib_d[matches] = model["width"] * (picked + 0.5)
        md = channel["daughter"]["mass"]
        mn = channel["neutral"]["mass"]
        mu = md * mn / (md + mn)
        angle = np.deg2rad(float(parameters["velocity_angle"]))
        velocity = float(parameters["speed"]) * np.array(
            [np.sin(angle), 0, np.cos(angle)]
        )
        relative = np.sqrt(2 * KB * k / mu)[:, None] * n
        vd = velocity + mn / (md + mn) * relative
        vn = velocity - md / (md + mn) * relative
        rd = np.sum(jd**2, axis=1) / (2 * id_ * KB)
        rn = (
            np.zeros(count) if model["atom"] else np.sum(jn**2, axis=1) / (2 * in_ * KB)
        )
        return dict(
            n=n,
            c=c,
            K=k,
            A=a,
            R=r,
            V=total_v,
            EvD=vib_d,
            EvN=total_v - vib_d,
            ErD=rd,
            ErN=rn,
            ell=ell,
            jd=jd,
            jn=jn,
            vd=vd,
            vn=vn,
            omega_d=jd / id_,
            J=j_parent,
            velocity=velocity,
            acceptance=accepted / proposed,
            proposed=proposed,
            remaining=remaining,
        )

    return angular_density, prepare_model, sample_outcomes


@app.cell(hide_code=True)
def _(channel, controls, default_parameters, mo, prepare_model, sample_outcomes):
    parameters = controls.value or default_parameters
    try:
        model = prepare_model(channel, parameters)
        outcomes = sample_outcomes(channel, model, parameters)
    except ValueError as _error:
        mo.stop(True, mo.callout(str(_error), kind="warn"))
    mo.md(f"""
    **{int(parameters["count"]):,} outcomes** · seed **{int(parameters["seed"])}** ·
    effective DOS width **{model["width"]:g} K** ({model["factor"]} stored bins per sampling bin).

    Parent: vibration **{model["threshold"] + model["q"] - model["rot"]:.2f} K**,
    rotation **{model["rot"]:.2f} K**, internal energy **{model["threshold"] + model["q"]:.2f} K**.
    Product excess: **{model["q"]:.2f} K**.
    Joint-draw acceptance: **{outcomes["acceptance"]:.1%}**;
    independently integrated expectation **{model["expected_acceptance"]:.1%}**.
    """)
    return model, outcomes, parameters


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numerical representation and an efficient joint draw

    The experiment reads each product's cached DOS **as stored**, including gaps
    and any omitted ground-state bin. A bin has mass $\rho_i\Delta$ at energy
    $(i+1/2)\Delta$. To bound work, adjacent bins may be merged; their masses are
    summed and placed at the merged midpoint. The effective width is reported.
    Increase the bin budget to check convergence, especially near threshold.

    We convolve the two *individual* product tables, so a molecular pair's total
    vibrational energies are $(i+j+1)\Delta$. Splitting that total uses precisely
    the same bin masses. This makes the sampled energy balance exact for the
    discretized model. It need not reproduce the separately cached `products_dos`
    table, which the simulator computes from concatenated frequencies. No missing
    ground state is silently invented. An atom-like neutral is explicitly a point
    mass at zero vibrational energy. Calculations below use log weights to handle
    the large DOS dynamic range.

    We sample $(V,\mathbf n)$ jointly without building a CDF for every direction.
    Put $c=\cos\theta=\mathbf n\cdot\widehat{\mathbf J}$,
    $C=J^2/(2I_\Sigma)$, $W_0=Q-V$ and $\nu=5/2$ ($\nu=1$ for an atom-like neutral).
    For each accessible $V$ define
    $$c_{\max}(V)=\min\{1,\sqrt{W_0/C}\},$$
    with $c_{\max}=1$ when $C=0$. Propose $V$ with weight
    $\rho_{DN}(V)W_0^\nu c_{\max}(V)$, then draw
    $c\sim\mathrm{Uniform}[-c_{\max},c_{\max}]$ and accept with probability
    $$(1-Cc^2/W_0)^\nu.$$
    The proposal's $c_{\max}$ cancels its conditional uniform density, leaving the
    required weight $\rho_{DN}(V)(Q-V-Cc^2)^\nu$. Draw the azimuth uniformly.
    At $J=0$ all proposals are accepted. The subsequent Dirichlet and angular draws
    conserve energy and both momenta for every outcome.
    """)
    return


@app.cell(hide_code=True)
def _(channel, mo, model, np, plt):
    input_figure, _axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained")
    for _role, _color in [
        ("parent", "#4338ca"),
        ("daughter", "#0f766e"),
        ("neutral", "#c2410c"),
    ]:
        _species = channel[_role]
        if _species["atom"]:
            continue
        _rho = _species["rho"]
        _x = channel["width"] * (np.arange(len(_rho)) + 0.5)
        _limit = (
            max(model["threshold"] + model["q"], model["q"])
            if _role == "parent"
            else model["q"]
        )
        _mask = (_x <= _limit) & np.isfinite(_rho) & (_rho > 0)
        if _mask.any():
            _axes[0].plot(
                _x[_mask], _rho[_mask] / _rho[_mask].max(), label=_role, color=_color
            )
    _axes[0].set(
        xlabel="Vibrational energy / kB (K)",
        ylabel="DOS / accessible peak",
        yscale="log",
        title="Stored individual DOS; separate scales",
    )
    _axes[0].legend()
    for _gid, _width, _data in channel["rates"]:
        _y = np.asarray(_data)
        _x = _width * (np.arange(len(_y)) + 0.5)
        _mask = np.isfinite(_y) & (_y > 0)
        _axes[1].plot(_x[_mask], _y[_mask], label=f"rate grid {_gid}: {_width:g} K")
    _rate_excess = model["threshold"] + model["q"] - channel["threshold"]
    _axes[1].axvline(
        _rate_excess, color="#475569", ls="--", label="chosen parent internal energy"
    )
    _axes[1].set(
        xlabel="(E − database D0) / kB (K)",
        ylabel="Stored rate (s⁻¹)",
        yscale="log",
        title="Context: rotation-averaged rate curves",
    )
    _axes[1].legend(fontsize=8)
    plt.close(input_figure)
    mo.vstack(
        [
            mo.md("## What the database supplies"),
            input_figure,
            mo.md(
                "Stored rate curves are shown for context only; their grid IDs are explicit. "
                "They are not fixed-J rates and do not drive this conditional sampler. "
                "The marker accounts for a threshold override; no rate is extrapolated."
            ),
        ]
    )
    return (input_figure,)


@app.cell(hide_code=True)
def _(angular_density, mo, model, np, outcomes, plt):
    distribution_figure, _axes = plt.subplots(
        2, 2, figsize=(10, 7), layout="constrained"
    )
    _edges = np.linspace(0, model["q"], 51)
    _mass, _ = np.histogram(model["v"], bins=_edges, weights=model["target_v"])
    _axes[0, 0].hist(
        outcomes["V"], bins=_edges, density=True, color="#99f6e4", label="samples"
    )
    _axes[0, 0].stairs(
        _mass / np.diff(_edges), _edges, color="#0f172a", label="integrated target"
    )
    _axes[0, 0].set(
        xlabel="Total product vibration V / kB (K)", ylabel="Probability density (K⁻¹)"
    )
    _bound = (
        1 if model["c_energy"] == 0 else min(1, np.sqrt(model["q"] / model["c_energy"]))
    )
    _cosines = np.linspace(-_bound, _bound, 501)
    _axes[0, 1].hist(
        outcomes["c"],
        bins=np.linspace(-_bound, _bound, 41),
        density=True,
        color="#c7d2fe",
        label="samples",
    )
    _axes[0, 1].plot(
        _cosines,
        angular_density(model, _cosines),
        color="#0f172a",
        label="integrated target",
    )
    _axes[0, 1].set(xlabel="cos θ = recoil direction · ẑ", ylabel="Probability density")
    for _key, _label, _color in [
        ("EvD", "daughter vibration", "#0f766e"),
        ("EvN", "neutral vibration", "#c2410c"),
    ]:
        _axes[1, 0].hist(
            outcomes[_key],
            bins=_edges,
            density=True,
            histtype="step",
            lw=2,
            color=_color,
            label=_label,
        )
    _axes[1, 0].set(
        xlabel="Individual vibrational energy / kB (K)",
        ylabel="Probability density (K⁻¹)",
    )
    for _key, _label, _color in [
        ("K", "relative recoil K", "#4338ca"),
        ("ErD", "daughter rotation", "#0f766e"),
        ("ErN", "neutral rotation", "#c2410c"),
    ]:
        _axes[1, 1].hist(
            outcomes[_key],
            bins=_edges,
            density=True,
            histtype="step",
            lw=2,
            color=_color,
            label=_label,
        )
    _axes[1, 1].set(
        xlabel="Physical mechanical energy / kB (K)", ylabel="Probability density (K⁻¹)"
    )
    for _ax in _axes.flat:
        _ax.legend(fontsize=8)
    plt.close(distribution_figure)
    mo.vstack(
        [
            mo.md("## Where the available energy goes"),
            distribution_figure,
            mo.md(
                "The black curves are calculated independently of the random draws. "
                "At zero parent rotation the recoil directions are isotropic: cos θ is uniform. "
                "For an atom-like neutral, its vibration and rotation are point masses at zero."
            ),
        ]
    )
    return (distribution_figure,)


@app.cell(hide_code=True)
def _(mo, np, outcomes, plt):
    state_figure, _axes = plt.subplots(1, 3, figsize=(11, 3.5), layout="constrained")
    _recoil = outcomes["vd"] - outcomes["velocity"]
    _stride = max(1, len(_recoil) // 3000)
    _scatter = _axes[0].scatter(
        _recoil[::_stride, 2],
        _recoil[::_stride, 0],
        c=outcomes["EvD"][::_stride],
        s=5,
        alpha=0.6,
        cmap="viridis",
    )
    _axes[0].set(
        xlabel="Daughter recoil Δvz (m/s)",
        ylabel="Daughter recoil Δvx (m/s)",
        aspect="equal",
    )
    state_figure.colorbar(_scatter, ax=_axes[0], label="Daughter vibration / kB (K)")
    _axes[1].hist(
        np.linalg.norm(outcomes["vd"], axis=1), bins=40, color="#0f766e", alpha=0.7
    )
    _axes[1].axvline(
        np.linalg.norm(outcomes["velocity"]),
        color="#4338ca",
        ls="--",
        label="parent speed",
    )
    _axes[1].set(xlabel="Daughter laboratory speed (m/s)", ylabel="Count")
    _axes[1].legend(fontsize=8)
    _axes[2].hist(
        np.linalg.norm(outcomes["omega_d"], axis=1) / 1e12,
        bins=40,
        color="#0f766e",
        alpha=0.7,
    )
    _axes[2].set(xlabel="Daughter |ω| (10¹² rad/s)", ylabel="Count")
    plt.close(state_figure)
    mo.vstack(
        [mo.md("## The state passed to the next trajectory segment"), state_figure]
    )
    return (state_figure,)


@app.cell(hide_code=True)
def _(KB, channel, mo, model, np, outcomes):
    _md = channel["daughter"]["mass"]
    _mn = channel["neutral"]["mass"]
    _mu = _md * _mn / (_md + _mn)
    _relative = outcomes["vd"] - outcomes["vn"]
    _reconstructed = (
        outcomes["EvD"]
        + outcomes["EvN"]
        + outcomes["ErD"]
        + outcomes["ErN"]
        + _mu * np.sum(_relative**2, axis=1) / (2 * KB)
    )
    _energy_error = np.max(np.abs(_reconstructed - model["q"])) / model["q"]
    _p_error = (
        _md * outcomes["vd"] + _mn * outcomes["vn"] - (_md + _mn) * outcomes["velocity"]
    )
    _p_scale = max(
        (_md + _mn) * np.linalg.norm(outcomes["velocity"]),
        np.sqrt(2 * _mu * KB * model["q"]),
    )
    _j_error = outcomes["jd"] + outcomes["jn"] + outcomes["ell"] - outcomes["J"]
    _j_scale = max(
        np.linalg.norm(outcomes["J"]),
        np.sqrt(2 * model["inertia_sum"] * KB * model["q"]),
    )
    conservation = {
        "energy / Q": float(_energy_error),
        "linear momentum / characteristic momentum": float(
            np.max(np.linalg.norm(_p_error, axis=1)) / _p_scale
        ),
        "angular momentum / characteristic angular momentum": float(
            np.max(np.linalg.norm(_j_error, axis=1)) / _j_scale
        ),
        "orbital ℓ · n / characteristic angular momentum": float(
            np.max(np.abs(np.sum(outcomes["ell"] * outcomes["n"], axis=1))) / _j_scale
        ),
    }
    mo.vstack(
        [
            mo.md("## Conservation checks on every sampled event"),
            mo.ui.table(
                [
                    {"maximum scaled residual": k, "value": v}
                    for k, v in conservation.items()
                ],
                selection=None,
                show_column_summaries=False,
            ),
            mo.callout(
                "All balances are within numerical precision."
                if max(conservation.values()) < 1e-10
                else "A balance exceeds 1e-10. Inspect the numerical regime before using these draws.",
                kind="success" if max(conservation.values()) < 1e-10 else "warn",
            ),
            mo.md(
                "Momentum scales stay nonzero at J = 0 or zero laboratory speed. "
                "The energy check reconstructs recoil from the actual velocities. "
                "Conservation validates reconstruction, not the physical accuracy of the statistical assumptions."
            ),
        ]
    )
    return (conservation,)


@app.cell(hide_code=True)
def _(mo, outcomes):
    event_index = mo.ui.slider(
        0,
        len(outcomes["K"]) - 1,
        step=1,
        value=0,
        label="Breakup to inspect",
        show_value=True,
    )
    mo.vstack([mo.md("## Inspect one breakup"), event_index])
    return (event_index,)


@app.cell(hide_code=True)
def _(channel, escape, event_index, mo, model, np, outcomes):
    _i = int(event_index.value)
    _colors = ["#0f766e", "#c2410c", "#4338ca"]
    _velocities = [outcomes["vd"][_i], outcomes["vn"][_i], outcomes["velocity"]]
    _vscale = 135 / max(1, *(np.linalg.norm(v[[2, 0]]) for v in _velocities))
    _vlines = []
    for _j, (_vector, _color) in enumerate(zip(_velocities, _colors, strict=True)):
        _end = np.array([220.0, 170.0]) + _vscale * _vector[[2, 0]] * [1, -1]
        _vlines.append(
            f'<path d="M220 170L{_end[0]:.2f} {_end[1]:.2f}" stroke="{_color}" stroke-width="3" fill="none" marker-end="url(#event-arrow-{_j})"/>'
        )
    _momenta = np.array([outcomes["jd"][_i], outcomes["jn"][_i], outcomes["ell"][_i]])
    _points = np.vstack([np.zeros(3), np.cumsum(_momenta, axis=0)])[:, [2, 0]] * [1, -1]
    _span = max(float(np.ptp(_points, axis=0).max()), 1e-40)
    _points = (_points - (_points.min(axis=0) + _points.max(axis=0)) / 2) * (
        230 / _span
    ) + [680, 170]
    _jlines = []
    for _j, _color in enumerate(_colors):
        _start, _end = _points[_j], _points[_j + 1]
        _jlines.append(
            f'<path d="M{_start[0]:.2f} {_start[1]:.2f}L{_end[0]:.2f} {_end[1]:.2f}" stroke="{_color}" stroke-width="3" fill="none" marker-end="url(#event-arrow-{_j})"/>'
        )
    _start, _end = _points[0], _points[-1]
    _jlines.append(
        f'<path d="M{_start[0]:.2f} {_start[1]:.2f}L{_end[0]:.2f} {_end[1]:.2f}" stroke="#0f172a" stroke-width="1.5" stroke-dasharray="5 4" fill="none"/>'
    )
    _defs = "".join(
        f'<marker id="event-arrow-{j}" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0 0L7 3.5L0 7" fill="{color}"/></marker>'
        for j, color in enumerate(_colors)
    )
    event_svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 920 380" role="img" aria-label="Selected breakup: projected velocities and a head-to-tail angular momentum balance">
    <defs>{_defs}</defs><rect width="920" height="380" rx="12" fill="#f8fafc"/>
    <g font-family="sans-serif" fill="#172554" font-size="14">
    <text x="30" y="30" font-size="18">Event {_i}: {escape(channel["parent"]["name"])}</text>
    <text x="220" y="58" text-anchor="middle">Laboratory velocities, z–x projection</text>
    <text x="680" y="58" text-anchor="middle">Angular momenta, head to tail</text>
    <path d="M60 170H390 M220 70V310" stroke="#cbd5e1"/><text x="392" y="175">+z</text><text x="227" y="82">+x</text>
    <circle cx="220" cy="170" r="4" fill="#475569"/>{"".join(_vlines)}{"".join(_jlines)}
    <text x="35" y="335" fill="#0f766e">daughter v_D</text><text x="177" y="335" fill="#c2410c">neutral v_N</text><text x="315" y="335" fill="#4338ca">parent V</text>
    <text x="500" y="335" fill="#0f766e">j_D</text><text x="565" y="335" fill="#c2410c">+ j_N</text><text x="647" y="335" fill="#4338ca">+ ℓ</text><text x="714" y="335">= J (dashed)</text>
    <text x="460" y="365" text-anchor="middle">Panels use independent scales. Projection can hide a vector; numerical values below include all three components.</text>
    </g></svg>"""
    _rows = [
        {
            "quantity": "velocity (m/s)",
            "daughter": np.array2string(outcomes["vd"][_i], precision=3),
            "neutral": np.array2string(outcomes["vn"][_i], precision=3),
        },
        {
            "quantity": "angular velocity (rad/s)",
            "daughter": np.array2string(outcomes["omega_d"][_i], precision=4),
            "neutral": "0 (atom-like)"
            if model["atom"]
            else np.array2string(
                outcomes["jn"][_i] / channel["neutral"]["inertia"], precision=4
            ),
        },
        {
            "quantity": "vibrational energy / kB (K)",
            "daughter": f"{outcomes['EvD'][_i]:.3f}",
            "neutral": f"{outcomes['EvN'][_i]:.3f}",
        },
        {
            "quantity": "rotational energy / kB (K)",
            "daughter": f"{outcomes['ErD'][_i]:.3f}",
            "neutral": f"{outcomes['ErN'][_i]:.3f}",
        },
    ]
    mo.vstack(
        [
            mo.Html(event_svg),
            mo.ui.table(_rows, selection=None, show_column_summaries=False),
            mo.md(
                f"Relative recoil energy: **{outcomes['K'][_i]:.3f} K**. "
                f"Product energies sum to **{(outcomes['K'][_i] + outcomes['V'][_i] + outcomes['ErD'][_i] + outcomes['ErN'][_i]):.3f} K** = Q/kB."
            ),
            mo.download(
                event_svg,
                filename="fragmentation-event.svg",
                mimetype="image/svg+xml",
                label="Download this SVG",
            ),
        ]
    )
    return (event_svg,)


@app.cell(hide_code=True)
def _(channel, csv, io, mo, model, np, outcomes, parameters):
    _columns = {
        "channel_id": np.full(len(outcomes["K"]), channel["channel_id"]),
        "dos_grid_id": np.full(len(outcomes["K"]), channel["grid_id"]),
        "seed": np.full(len(outcomes["K"]), int(parameters["seed"])),
        "Q_K": np.full(len(outcomes["K"]), model["q"]),
        "D0_K": np.full(len(outcomes["K"]), model["threshold"]),
        "parent_rotation_K": np.full(len(outcomes["K"]), model["rot"]),
        "sampling_bin_width_K": np.full(len(outcomes["K"]), model["width"]),
    }
    for _key in ["K", "EvD", "EvN", "ErD", "ErN"]:
        _columns[f"{_key}_K"] = outcomes[_key]
    for _key, _unit in [
        ("vd", "m_s"),
        ("vn", "m_s"),
        ("omega_d", "rad_s"),
        ("jd", "kg_m2_s"),
        ("jn", "kg_m2_s"),
        ("ell", "kg_m2_s"),
    ]:
        for _axis, _coordinate in enumerate("xyz"):
            _columns[f"{_key}_{_coordinate}_{_unit}"] = outcomes[_key][:, _axis]
    _buffer = io.StringIO()
    _writer = csv.writer(_buffer)
    _writer.writerow(_columns)
    _writer.writerows(zip(*_columns.values(), strict=True))
    mo.download(
        _buffer.getvalue(),
        filename="fragmentation-outcomes.csv",
        mimetype="text/csv",
        label="Download all sampled outcomes (CSV)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## What this experiment establishes — and what still needs a model choice

    * **Rates:** the paper's Eq. 21 averages over parent rotational energy $r$:
      $\bar k(E)=\int f_r(r\mid E)k_0(E-D_0-r)\,dr$, where
      $f_r(r\mid E)\propto\rho_P(E-r)\sqrt r$ on $0\le r\le E$.
      A fragmenting ensemble is rate biased:
      $p(r\mid\mathrm{event},E)\propto f_r(r\mid E)k_0(E-D_0-r)$.
      The sampler here retains the supplied $\mathbf J$. Combining it with existing
      rotation-averaged event times is a hybrid approximation. A consistent
      fixed-$J$ kinetics model would also need $k(E,J)$ from the same flux measure
      and a parent DOS at fixed $J$.
    * **Capture:** $b_{\max}=\infty$ is the approximation used in the paper's Eq. 39.
      To impose a finite capture law, reconstruct $b=|\boldsymbol\ell|/\sqrt{2\mu K}$
      and reject entire draws outside the capture domain. That changes the target
      plots as well. No finite capture law is assumed or implemented here.
      Agreement of integrated rates in the paper's example does not validate all
      product spectra or all channels.
    * **Rotation:** the parent $\mathbf J=I_P\boldsymbol\omega_P$ is the complete
      angular momentum only within this spherical model. Vibrational angular
      momentum and anisotropic rotors are not represented. The shared simulator's
      rotational–vibrational redistribution already makes an approximation here.
    * **DOS and thresholds:** the input tables and their ground-state conventions
      matter. The discrete sampler is exact for its chosen midpoint masses;
      physical and continuum accuracy still require a resolution study and
      consistent zero-point-corrected thresholds. Empty accessible support stops
      sampling rather than manufacturing a zero-energy outcome.

    ### Transcription checks and references

    The original analysis visually checked printed pp. 615–619. Eq. 23 prints
    $E-\epsilon_r$ in its last line; the surrounding definitions and Eqs. 22,
    35–39 instead require $E-D_0-\epsilon_r=\varepsilon-\epsilon_r$. Eq. 39 uses
    $\widetilde\varepsilon_t\sqrt{\widetilde\varepsilon_r}$, and the rational
    impact-parameter factor in Eq. 38 is **squared**. Those distinctions are
    retained in the derivation above.

    * [Zapadinsky et al., J. Phys. Chem. A 123, 611–624 (2019)](https://doi.org/10.1021/acs.jpca.8b10744).
    * [Calvo and Parneix, Statistical evaporation of rotating clusters. III. Molecular clusters (2004)](https://doi.org/10.1063/1.1633258)
      develops angular-momentum-resolved phase-space descriptions beyond the
      zero-$J$ calculation. The fixed-$J$ construction above is derived here,
      not quoted from that reference.

    Run with the notebook's declared dependencies:
    ```bash
    uvx marimo edit --sandbox examples/notebooks/fragmentation-outcome-sampler.py
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
