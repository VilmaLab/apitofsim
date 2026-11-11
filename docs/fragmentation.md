# Daughter states after fragmentation

Cluster-tree simulations can continue with a sampled charged daughter:

```toml
[default_config.simulation]
simulation_mode = "cluster-tree"
fragmentation_state_method = "phase-space"
```

The default, `fragmentation_state_method = "zero"`, retains the previous reset of
velocity, angular velocity and vibrational energy. The setting is saved with the
experiment configuration. It only affects continuing tree transitions; terminal
fragmentations and the single-cluster/pathway-at-a-time modes retain their behavior.
`postfragmentation` events record the assigned state using the existing schema.

## Model and limits

The sampler extends the differential flux in Eqs. 35–39 of
[Zapadinsky et al. (2019), *Modeling on Fragmentation of Clusters inside a Mass Spectrometer*](https://doi.org/10.1021/acs.jpca.8b10744)
to the simulated parent's instantaneous angular momentum. It treats each molecular
product as a spherical rotor, builds vibrational densities of states (DOS) from
the existing product frequencies, and assumes an unlimited capture impact
parameter, as in Eq. 39.
The neutral product is sampled for conservation but is not propagated.

This is an experimental continuation model. Event times and channel probabilities
still use the existing rotation-averaged rates, not rates resolved in angular
momentum. Consequently the combined model is an approximation, even though each
sample conserves energy and both momenta. The paper's agreement of integrated
rates with a finite capture model does not validate all daughter-state distributions.
The sampled velocities describe separated products, assigned instantaneously at
the parent's position and time; molecular separation and torques are not resolved.

## Distribution

The [mathematical reference](simulation-methods.pdf) gives the full flux derivation.
The implemented sampling recipe is summarized here.

For parent velocity $\mathbf V$, angular momentum $\mathbf J=I_P\boldsymbol\omega_P$,
and internal energy $E$, the available product energy is $Q=E-D_0$. The same
channel threshold $D_0$ used by the rate calculation is subtracted once. Parent
laboratory kinetic energy is excluded from $Q$. Product masses must sum to the
parent mass, and the sampled daughter's inertia must match its tree payload.

Write $I_\Sigma=I_D+I_N$, $I_*=I_DI_N/I_\Sigma$ and
$\mu=m_Dm_N/(m_D+m_N)$. For recoil direction $\mathbf n$, define

$$
Q_{\mathbf n}=Q-\frac{(\mathbf J\cdot\mathbf n)^2}{2I_\Sigma},
\qquad
\rho_{DN}(V)=\int_0^V\rho_D(u)\rho_N(V-u)\,du.
$$

The joint density of direction and total vibrational energy is

$$
p(\mathbf n,V)\,d\Omega\,dV\ \propto\
\rho_{DN}(V)(Q_{\mathbf n}-V)^{5/2}\,d\Omega\,dV,
\qquad 0\le V<Q_{\mathbf n}.
$$

Conditional on these, divide $W=Q_{\mathbf n}-V$ as
$(K,A,R)/W\sim\operatorname{Dirichlet}(1,1,3/2)$.
Draw $\mathbf e$ uniformly on the circle perpendicular to $\mathbf n$ and
$\mathbf a$ uniformly on the sphere. Reconstruct

$$
\begin{aligned}
\boldsymbol\ell&=\mathbf J-(\mathbf J\cdot\mathbf n)\mathbf n
                  +\sqrt{2I_\Sigma A}\,\mathbf e,\\
\mathbf s&=\sqrt{2I_*R}\,\mathbf a,\\
\mathbf j_D&=\frac{I_D}{I_\Sigma}(\mathbf J-\boldsymbol\ell)+\mathbf s,\\
\mathbf j_N&=\frac{I_N}{I_\Sigma}(\mathbf J-\boldsymbol\ell)-\mathbf s.
\end{aligned}
$$

Sample the vibrational split with weight $\rho_D(u)\rho_N(V-u)$. The daughter receives

$$
\mathbf v_D=\mathbf V+\frac{m_N}{m_D+m_N}\sqrt{\frac{2K}{\mu}}\,\mathbf n,
\qquad \boldsymbol\omega_D=\mathbf j_D/I_D,
\qquad E_{v,D}=u.
$$

These obey $\mathbf j_D+\mathbf j_N+\boldsymbol\ell=\mathbf J$ and
$K+j_D^2/(2I_D)+j_N^2/(2I_N)+V=Q$, as well as linear momentum conservation.
For zero parent angular momentum, integrating out $K$ at fixed
$\widetilde K=K+A$ gives the paper's kernel
$\widetilde K\sqrt R\,\rho_{DN}(Q-\widetilde K-R)$.
The transformed energy $\widetilde K$ includes rotational energy and must not be
used directly as recoil energy.

For an atom-like neutral, its spin and vibrational energy are zero. Omit $R$ and
$\mathbf s$, use exponent $1$ instead of $5/2$, and use
$(K,A)/W\sim\operatorname{Dirichlet}(1,1)$ with $I_\Sigma=I_D$.

## Numerical implementation and API

The sampler builds ground-inclusive Beyer–Swinehart counts on the existing DOS
grid, rounding each oscillator frequency to the nearest bin. Frequencies that
round to zero are rejected: reduce `bin_width` to resolve those modes. Counts at
excitation $i\,\Delta E$ are summed rather than integrated, so the ground state is
accessible even with less than one bin of available energy. Individual and combined
counts use the same recurrence, ensuring consistent vibrational splits.

The current rate DOS routine omits the all-ground-state contribution; the sampler
therefore does not reuse its values. This setting changes neither those tables nor
the rates computed from them. Collision sampling also retains its existing
bin-centre convention. Refine `bin_width` when assessing convergence. All runtime
energies are joules. Counts are stored in logarithms and sampling weights are
evaluated in log space. DOS preparation switches to log space before counts overflow.
Out-of-range energies or missing/invalid product data raise errors; they do not
silently fall back to zeroing or clip away energy.

Workflow preparation supplies both products' frequencies and the actual prepared
DOS grid, including when loading cached rates. Prepared product data are immutable
and shared across realizations. The per-realization random generator controls all
sampling, and the rejection loop checks cooperative cancellation.

For direct Python use, construct `api.FragmentationProducts(daughter, neutral,
bin_width, energy_max)` with `ClusterData` objects and Pint energy quantities
(including kelvin), or bare joules. Add `products=...` to
`api.MassSpecInputFragmentationPathway`, and set
`fragmentation_state_method=api.FragmentationStateMethod.phase_space` on
`api.MassSpecSubstanceTreeInput`. Existing calls retain the zeroing default.
