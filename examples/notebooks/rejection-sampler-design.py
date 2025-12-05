# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib==3.10.7",
#     "numpy==2.3.5",
#     "scipy==1.16.3",
#     "sympy==1.14.0",
# ]
# ///

import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Prelude

    We start by importing some packages and defining some constants.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    from math import exp, pi, sqrt, cos, sin
    import numpy as np
    from numpy.linalg import norm
    from matplotlib import pyplot as plt
    return cos, exp, mo, norm, np, pi, plt, sin, sqrt


@app.cell(hide_code=True)
def _(np, sqrt):
    # Init constants

    rng = np.random.default_rng(42)
    amu = 216
    boltzmann = 1.38064852e-23
    kT = boltzmann * 300
    proton_mass = 1.6726219e-27
    m_ion = amu * proton_mass

    gas_molecule_mass_kg = 4.8506e-26
    mobility_gas_inv = gas_molecule_mass_kg / kT
    mobility_gas = kT / gas_molecule_mass_kg
    boundary_u = 5.0 * sqrt(mobility_gas)
    return (
        amu,
        boltzmann,
        boundary_u,
        gas_molecule_mass_kg,
        kT,
        m_ion,
        mobility_gas,
        mobility_gas_inv,
        proton_mass,
        rng,
    )


@app.cell(hide_code=True)
def _(
    amu,
    boltzmann,
    boundary_u,
    gas_molecule_mass_kg,
    kT,
    m_ion,
    mobility_gas,
    mobility_gas_inv,
    proton_mass,
):
    constants = {
        "amu": amu,
        "boltzmann": boltzmann,
        "kT": kT,
        "proton_mass": proton_mass,
        "m_ion": m_ion,
        "gas_molecule_mass_kg": gas_molecule_mass_kg,
        "mobility_gas_inv": mobility_gas_inv,
        "mobility_gas": mobility_gas,
        "boundary_u": boundary_u
    }

    for key, value in constants.items():
        print(f"{key}: {value}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we look at the typical initial range for the value of v_norm, and get a reasonable default and maximum value for the sliders later on.
    """)
    return


@app.cell
def _(kT, m_ion, norm, plt, rng, sqrt):
    def gen_v_norm(rng):
        v = sqrt(kT / m_ion) * rng.standard_normal(3)
        return norm(v)

    v_norm_values = sorted([gen_v_norm(rng) for _ in range(100000)])
    v_norm_max = v_norm_values[-100]
    v_norm_median = v_norm_values[len(v_norm_values) // 2]
    v_norm_dist = plt.hist(v_norm_values, bins="auto")[2]
    v_norm_dist
    return v_norm_max, v_norm_median


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Exploratory plotting

    The notation here is a mix of this publication

    * Zapadinsky, E., Passananti, M., Myllys, N., Kurtén, T., & Vehkamäki, H. (2019). Supporting Information to "Modelling on Fragmentation of Clusters Inside a Mass Spectrometer" [[pdf]](https://pubs.acs.org/doi/suppl/10.1021/acs.jpca.8b10744/suppl_file/jp8b10744_si_001.pdf)

    and conventions/variable names from the main apitofsim source code.

    In the next section we draw a few plots of $\Upsilon$. This plotting was used while writing the rejection sampler. If you are only interested in the rejection sampler itself, you can mostly ignore this section.

    We start with the following functions, which model $\Upsilon$, but drop all constant factors. The first function is $\propto \Upsilon(\theta | u_n)$ while the second is $\propto \Upsilon(\theta, u_n)$.

    It's convenient to work in a rectangular domain, so we define $u = u_n + v cos(\theta)$ and define two more functions which are $\propto \Upsilon(\theta | u)$ and $\propto \Upsilon(\theta, u)$ respectively.
    """)
    return


@app.cell
def _(cos, exp, sin):
    def func_theta(u_n, theta, v_norm, mobility_gas_inv):
        return (u_n + v_norm * cos(theta)) * exp(-0.5 * mobility_gas_inv * u_n ** 2)

    def func_joint(u_n, theta, v_norm, mobility_gas_inv):
        return (u_n + v_norm * cos(theta)) * exp(-0.5 * mobility_gas_inv * u_n ** 2) * sin(theta)

    def func_theta0(u, theta, v_norm, mobility_gas_inv):
        return u * exp(-0.5 * mobility_gas_inv * (u - v_norm * cos(theta)) ** 2)

    def func_joint0(u, theta, v_norm, mobility_gas_inv):
        return u * exp(-0.5 * mobility_gas_inv * (u - v_norm * cos(theta)) ** 2) * sin(theta)
    return func_joint, func_joint0, func_theta, func_theta0


@app.cell(hide_code=True)
def _(mo, pi, v_norm_max, v_norm_median):
    theta_w = mo.ui.slider(start=0, stop=pi, step=pi / 1000)
    v_norm_w = mo.ui.slider(start=0, stop=v_norm_max, step=v_norm_max / 1000, value=v_norm_median)
    zero_start_w = mo.ui.checkbox(label="Zero start", value=False)
    return theta_w, v_norm_w, zero_start_w


@app.cell(hide_code=True)
def _(boundary_u, mo, theta_w, v_norm_w, zero_start_w):
    u_w = mo.ui.slider(start=0 if zero_start_w.value else -v_norm_w.value, stop=boundary_u, step=(boundary_u + v_norm_w.value) / 1000)
    d = mo.ui.dictionary(
        {
            "v_norm": v_norm_w,
            "theta": theta_w,
            "u": u_w,
            "zero_start": zero_start_w,
            "joint": mo.ui.checkbox(label="Joint", value=False),
        }
    )
    return (d,)


@app.cell(hide_code=True)
def _(boundary_u, d, func_joint, func_joint0, func_theta, func_theta0, np, pi):
    if d["joint"].value:
        if d["zero_start"].value:
            func = func_joint0
        else:
            func = func_joint
    else:
        if d["zero_start"].value:
            func = func_theta0
        else:
            func = func_theta
    func_v = np.vectorize(func)
    if d["zero_start"].value:
        u_ns = np.linspace(0, boundary_u + d["v_norm"].value, 100)
    else:
        u_ns = np.linspace(-d["v_norm"].value, boundary_u, 100)

    thetas = np.linspace(0, pi, 100)
    return func_v, thetas, u_ns


@app.cell(hide_code=True)
def _(cos, d, func_v, mobility_gas_inv, np, plt, thetas, u_ns):
    def mk_plt2d():
        u_ns_grid, thetas_grid = np.meshgrid(u_ns, thetas)
        data2d = func_v(u_ns_grid, thetas_grid, d["v_norm"].value, mobility_gas_inv)
        masked_data = np.ma.masked_where(data2d < 0, data2d)
        plt2d, ax = plt.subplots()
        im = ax.contourf(u_ns, thetas, masked_data, vmin=0, levels=50, cmap='viridis')
        ax.axhline(d["theta"].value, color='red')
        ax.axvline([d["u"].value], color='red')
        if not d["zero_start"].value:
            ax.plot(-d["v_norm"].value * np.vectorize(cos)(thetas), thetas)
        plt2d.colorbar(im)
        return plt2d

    plt2d = mk_plt2d()
    return (plt2d,)


@app.cell(hide_code=True)
def _(cos, d, func_v, mobility_gas_inv, plt, u_ns):
    def mk_plttheta():
        plttheta, ax = plt.subplots()
        ax.plot(u_ns, func_v(u_ns, d["theta"].value, d["v_norm"].value, mobility_gas_inv))
        if not d["zero_start"].value:
            ax.axvline(-d["v_norm"].value * cos(d["theta"].value), color='red')
            ax.axhline(0, color='red')
        return plttheta

    plttheta = mk_plttheta()
    return (plttheta,)


@app.cell(hide_code=True)
def _(d, func_v, mobility_gas_inv, plt, thetas):
    pltv = plt.plot(thetas, func_v(d["u"].value, thetas, d["v_norm"].value, mobility_gas_inv))
    return (pltv,)


@app.cell
def _(Finally):
    Finally, 
    return


@app.cell(hide_code=True)
def _(d, mo, plt2d, plttheta, pltv):
    mo.vstack([mo.hstack([d, plt2d]), mo.hstack([plttheta, pltv])], align="center")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Rejection sampler

    Next we'll design a (very simple) rejection sampler. This is a common technique, and so there are a lot of references, but I used the notes from Matti Vihola's course on Stochastic Simulation namely [notes section 2](https://users.jyu.fi/~mvihola/stochsim/notes2.pdf) and [notes section 3](https://users.jyu.fi/~mvihola/stochsim/notes3.pdf).

    The rejection sampler samples $u$ and $\theta$ from the joint probability distribution `func_joint0` using a uniform proposal distribution.
    """)
    return


@app.cell(hide_code=True)
def _():
    import sympy as sp
    sp.init_printing()
    return (sp,)


@app.cell
def _(mo):
    mo.md(r"""
    We note that $\Upsilon(\theta, u)$/`func_joint0` is monotonically increasing in the values of the terms $sin(\theta)$ and $cos(\theta)$ and these are both bounded at $1$ so we can create a bound by setting these to 1.
    """)
    return


@app.cell(hide_code=True)
def _(mobility_gas_inv, sp):
    def bound_function(u, v, g = mobility_gas_inv):
        return u * sp.exp(-g / 2 * (u - v) ** 2)

    u_s, v_s, g_s = sp.symbols("u v g_inv", positive=True, real=True)
    bound_function(u_s, v_s, g_s)
    return bound_function, g_s, u_s, v_s


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can plot it to have a look at the situation for values we were plotting above:
    """)
    return


@app.cell(hide_code=True)
def _(bound_function, d, np, plt, u_ns):
    plt.plot(u_ns, np.vectorize(bound_function)(u_ns, d["v_norm"].value))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Next we'll find the point where it reachest its maximum using calculus, passing this back into the bound function gives us an upper found on the value of `func_joint0`
    """)
    return


@app.cell
def _(bound_function, g_s, sp, u_s, v_s):
    max_point_s = sp.simplify(sp.solve(sp.diff(bound_function(u_s, v_s, g_s), u_s), u_s)[1])
    max_point_s
    return (max_point_s,)


@app.cell
def _(d, g_s, max_point_s, mobility_gas_inv, v_s):
    max_u = float(max_point_s.subs({v_s: d["v_norm"].value, g_s: mobility_gas_inv}))
    max_u
    return (max_u,)


@app.cell
def _(d, mobility_gas, sqrt):
    def bound_func_argmax(v):
        return (v + sqrt(v ** 2 + 4 * mobility_gas)) / 2

    bound_func_argmax(d["v_norm"].value)
    return


@app.cell
def _(bound_function, d, max_u):
    bound = float(bound_function(max_u, d["v_norm"].value))
    bound
    return (bound,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we we have an upper bound on the value of `func_joint0`, we can run a rejection sampler, for the same parameters chosen above:
    """)
    return


@app.cell
def _(bound, boundary_u, cos, d, func_joint0, mobility_gas_inv, np, pi, rng):
    def sample_2d(rng):
        rejections = 0
        while 1:
            u = rng.uniform(0, boundary_u + d["v_norm"].value)
            theta = rng.uniform(0, pi)
            pdf_val = func_joint0(u, theta, d["v_norm"].value, mobility_gas_inv)
            if rng.uniform(0, bound) < pdf_val:
                return rejections, u, theta
            rejections += 1

    def run_sampling(convert=False):
        n_samples = 100000
        samples = []
        sample_rejections = []
        for _ in range(n_samples):
            rejections, u_smpl, theta_smpl = sample_2d(rng)
            if convert:
                u_smpl -= d["v_norm"].value * cos(theta_smpl)
            samples.append((u_smpl, theta_smpl))
            sample_rejections.append(rejections)
        return sample_rejections, np.array(samples)

    rejections, samples = run_sampling();
    return rejections, run_sampling, samples


@app.cell
def _(plt, samples):
    plt.hist2d(samples[:, 0], samples[:, 1], bins=[50, 50])[-1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot shows the number of rejections. While it's moderately high, the functions involved are quite cheap, so sampling is fairly fast.
    """)
    return


@app.cell
def _(plt, rejections):
    plt.hist(rejections, bins="auto")[-1]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot shows the samples converted back to $u_n$ values.
    """)
    return


@app.cell
def _(plt, run_sampling):
    _, samples_c = run_sampling(True)
    plt.hist2d(samples_c[:, 0], samples_c[:, 1], bins=[50, 50])[-1]
    return


if __name__ == "__main__":
    app.run()
