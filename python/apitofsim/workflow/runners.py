from typing import Callable, Optional, Tuple

import numpy
from pint import get_application_registry

from apitofsim.api import (
    ApiTofError,
    ApiTofOverflowError,
    MassSpecInputFragmentationPathway,
    MassSpecSubstanceSingleInput,
    MassSpecSubstanceTreeInput,
    MassSpectrometer,
    MeshMode,
    SampleMode,
)

from .base import SimulationMode
from .db import (
    EventRecorder,
    ExperimentDatabase,
    RealizationDatabase,
    SuperClusterDatabase,
)
from .db_utils import get_or_insert, insert_via_arrow

ureg = get_application_registry()
Q_ = ureg.Quantity


def counter_fragmented_total(counters):
    return int(counters.n_fragmented_total.sum())


class DerivedDataPreparer:
    def __init__(self, db: SuperClusterDatabase):
        self.db = db

    def _run_skimmer(self, config):
        from apitofsim import skimmer

        return skimmer(
            T0=config["T"],  # pyright: ignore[reportCallIssue]
            P0=config["pressures"][0],
            rmax=config["lengths"][-1],
            dc=config["dc"],
            alpha_factor=config["alpha_factor"],
            gas=config["gas"],
            N=config["N_iter"],
            M=config["M_iter"],
            resolution=config["resolution"],
            tolerance=config["tolerance"],
        )

    def _get_cached_dos(
        self,
        cluster_indexed,
        histogram_id,
        pathway_lookup,
        include_cluster_dos=True,
        include_product_dos=True,
    ):
        num_misses = 0
        cluster_dos_dict = {}
        if include_cluster_dos:
            for cluster_id, cluster in self.db.get_cluster_dos_through_join_else(
                cluster_indexed,
                histogram_id,
                cluster_dos_dict,
            ):
                num_misses += 1

        if include_product_dos:
            for product_cluster in self.db.get_product_dos_through_join_else(
                cluster_indexed,
                histogram_id,
                cluster_dos_dict,
                pathway_lookup,
            ):
                num_misses += 1

        if num_misses > 0:
            raise ValueError(
                f"Expected to get all cluster DOS from the database, but got {num_misses} misses"
            )

        return cluster_dos_dict

    def _run_dos(
        self,
        cluster_indexed,
        histogram_id,
        bin_width,
        energy_max,
        pathway_lookup,
        status_table=None,
    ):
        import pyarrow as pa

        from apitofsim.api import compute_density_of_states_batch

        num_clusters_missed = 0
        cluster_dos_dict = {}
        missed_cluster_ids = []
        density_of_states_inputs = []
        for cluster_id, cluster in self.db.get_cluster_dos_through_join_else(
            cluster_indexed,
            histogram_id,
            cluster_dos_dict,
        ):
            missed_cluster_ids.append(cluster_id)
            density_of_states_inputs.append(cluster)
            num_clusters_missed += 1

        wanted_p1 = []
        wanted_p2 = []
        for _, product1_id, product2_id in pathway_lookup.values():
            product1_id, product2_id = sorted((product1_id, product2_id))
            wanted_p1.append(product1_id)
            wanted_p2.append(product2_id)

        density_of_states_inputs.extend(
            self.db.get_product_dos_through_join_else(
                cluster_indexed, histogram_id, cluster_dos_dict, pathway_lookup
            )
        )

        if status_table is not None:
            status_table["Description"] = (
                f"{len(cluster_dos_dict)} retrieved; {len(density_of_states_inputs)} to compute"
            )

        if len(density_of_states_inputs) > 0:
            density_of_states = compute_density_of_states_batch(
                density_of_states_inputs,
                energy_max=energy_max,
                bin_width=bin_width,
            )
            assert density_of_states.flags.f_contiguous
            if num_clusters_missed > 0:
                arrow_arr = pa.FixedSizeListArray.from_arrays(
                    density_of_states[:, :num_clusters_missed].flatten("K"),
                    density_of_states.shape[0],
                )

                insert_via_arrow(
                    self.db.db,
                    "cluster_dos",
                    histogram_params_id=[histogram_id] * len(missed_cluster_ids),
                    cluster_id=missed_cluster_ids,
                    data=arrow_arr,
                )

            if len(density_of_states_inputs) > num_clusters_missed:
                arrow_arr = pa.FixedSizeListArray.from_arrays(
                    density_of_states[:, num_clusters_missed:].flatten("K"),
                    density_of_states.shape[0],
                )
                insert_via_arrow(
                    self.db.db,
                    "products_dos",
                    histogram_params_id=[histogram_id] * len(wanted_p1),
                    cluster1_id=wanted_p1,
                    cluster2_id=wanted_p2,
                    data=arrow_arr,
                )

            for cluster_id, v in zip(
                missed_cluster_ids, density_of_states[:, : len(missed_cluster_ids)].T
            ):
                cluster_dos_dict[cluster_id] = v

            for p1, p2, v in zip(
                wanted_p1, wanted_p2, density_of_states[:, len(missed_cluster_ids) :].T
            ):
                cluster_dos_dict[(p1, p2)] = v
        return cluster_dos_dict

    def _run_k_total(
        self,
        cluster_indexed,
        cluster_dos_dict,
        mesh,
        bin_width,
        energy_max,
        energy_max_rate,
        pathway_lookup,
        status_table=None,
        name_lookup=None,
    ):
        import pyarrow as pa

        from apitofsim import (
            FragmentationPathway,
            KTotalInput,
            compute_k_total_batch,
            consts,
        )
        from apitofsim.api import validate_max_energies

        histogram_id = get_or_insert(
            self.db.db,
            "histogram_params",
            bin_width=bin_width.to("K").magnitude,
            max=energy_max_rate.to("K").magnitude,
        )

        k_total_inputs = []
        k_total_keys = []
        k_total_dict = {}

        for (
            pathway_id,
            cluster_id,
            product1_id,
            product2_id,
        ) in self.db.get_k_rate_through_join_else(
            histogram_id, k_total_dict, pathway_lookup
        ):
            parent_cpp = cluster_indexed[cluster_id].into_cpp()
            product1_cpp = cluster_indexed[product1_id].into_cpp()
            product2_cpp = cluster_indexed[product2_id].into_cpp()
            fragmentation_energy = FragmentationPathway(
                parent_cpp, product1_cpp, product2_cpp
            ).fragmentation_energy_kelvin()
            try:
                validate_max_energies(
                    fragmentation_energy=fragmentation_energy,
                    bin_width=bin_width,
                    energy_max=energy_max,
                    energy_max_rate=energy_max_rate,
                    quantities_strict=False,
                )
            except ApiTofError as e:
                if name_lookup is not None:
                    parent_name = name_lookup[cluster_id]
                    product1_name = name_lookup[product1_id]
                    product2_name = name_lookup[product2_id]
                    e.add_note(
                        f"Pathway is {parent_name} -> {product1_name} + {product2_name}"
                    )
                    e.add_note(
                        f"{parent_name}: {parent_cpp.electronic_energy * consts.hartK}K"
                    )
                    e.add_note(
                        f"{product1_name}: {product1_cpp.electronic_energy * consts.hartK}K"
                    )
                    e.add_note(
                        f"{product2_name}: {product2_cpp.electronic_energy * consts.hartK}K"
                    )
                raise e
            product_id_tpl = tuple(sorted((product1_id, product2_id)))
            k_total_keys.append(pathway_id)
            assert cluster_dos_dict[cluster_id].flags.c_contiguous
            assert cluster_dos_dict[product_id_tpl].flags.c_contiguous
            k_total_inputs.append(
                KTotalInput(
                    product1_cpp,
                    product2_cpp,
                    fragmentation_energy,
                    cluster_dos_dict[cluster_id],
                    cluster_dos_dict[product_id_tpl],
                )
            )

        progress_callback: Callable[[int], None] | None = None
        if status_table is not None:
            status_table["Description"] = (
                f"{len(k_total_dict)} retrieved; {len(k_total_inputs)} to compute"
            )

            inner_pbar = status_table(
                len(k_total_inputs),
                position=1,
                description="K total",
                show_throughput=False,
                show_progress=True,
                show_eta=True,
            )

            def _progress_callback(iters_done):
                inner_pbar.set_step(iters_done)

            progress_callback = _progress_callback

        k_rates = compute_k_total_batch(
            k_total_inputs,
            energy_max_rate=energy_max_rate,
            bin_width=bin_width,
            mesh=mesh,
            progress_callback=progress_callback,
        )

        insert_via_arrow(
            self.db.db,
            "k_rate",
            histogram_params_id=[histogram_id] * k_rates.shape[1],
            pathway_id=k_total_keys,
            data=pa.FixedSizeListArray.from_arrays(
                k_rates.flatten("K"), k_rates.shape[0]
            ),
        )
        for key, k_rates in zip(k_total_keys, k_rates.T):
            k_total_dict[key] = k_rates

        return k_total_dict

    def run_densityandrate(
        self, config, cluster_indexed, pathway_lookup, tablepbar=None, name_lookup=None
    ):
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        from apitofsim import precompute_mesh

        if tablepbar is None:
            table = ProgressTable(default_column_alignment="left", refresh_rate=0)
            total_steps = 3
            cur_step = 1
            pbar = table(
                3,
                description="Preliminary steps",
                show_throughput=False,
                show_progress=True,
                position=2,
            )
            table.add_column("#", alignment="right", width=3)
            table.add_column("Step", width=20)
            table.add_column("Description", width=40)
            table.add_column("Time (s)", alignment="right")
        else:
            table, pbar = tablepbar
            total_steps = 4
            cur_step = 2
        table["#"] = f"{cur_step}/{total_steps}"
        table["Step"] = "Density of states"
        pbar.update()
        start = timer()

        histogram_id = get_or_insert(
            self.db.db,
            "histogram_params",
            bin_width=config["bin_width"].to("K").magnitude,
            max=config["energy_max"].to("K").magnitude,
        )

        cluster_dos_dict = self._run_dos(
            cluster_indexed,
            histogram_id,
            config["bin_width"],
            config["energy_max"],
            pathway_lookup=pathway_lookup,
            status_table=table,
        )

        table["Time (s)"] = f"{(timer() - start):.2f}"
        table.next_row()
        cur_step += 1

        table["#"] = f"{cur_step}/{total_steps}"
        table["Step"] = "Computing mesh"
        pbar.update()
        start = timer()

        mesh_points = config["energy_max_rate"] / config["bin_width"]
        table["Description"] = f"Mesh of {mesh_points} pts"

        mesh = precompute_mesh(
            energy_max_rate=config["energy_max_rate"],
            bin_width=config["bin_width"],
            mesh_mode=MeshMode.compute_mesh_diagonal_multithreaded,
        )
        table["Time (s)"] = f"{(timer() - start):.2f}"
        table.next_row()
        cur_step += 1

        table["#"] = f"{cur_step}/{total_steps}"
        table["Step"] = "K total"
        pbar.update()
        start = timer()

        k_rates = self._run_k_total(
            cluster_indexed,
            cluster_dos_dict,
            mesh,
            config["bin_width"],
            config["energy_max"],
            config["energy_max_rate"],
            pathway_lookup=pathway_lookup,
            status_table=table,
            name_lookup=name_lookup,
        )

        table["Time (s)"] = f"{(timer() - start):.2f}"
        table.close()
        if pbar._is_active:
            # Not sure why this is sometimes needed
            pbar.close()

        return k_rates, cluster_dos_dict

    def get_cached_densityandrate(
        self, cluster_indexed, dos_histogram_id, rate_histogram_id, pathway_lookup
    ):
        num_missed = 0
        k_rates = {}
        for (
            pathway_id,
            cluster_id,
            product1_id,
            product2_id,
        ) in self.db.get_k_rate_through_join_else(
            rate_histogram_id, k_rates, pathway_lookup
        ):
            num_missed += 1

        if num_missed > 0:
            raise ValueError(
                f"Expected to get all k rates from the database, but got {num_missed} misses"
            )

        cluster_dos_dict = self._get_cached_dos(
            cluster_indexed,
            dos_histogram_id,
            pathway_lookup,
            include_cluster_dos=True,
            include_product_dos=False,
        )
        return k_rates, cluster_dos_dict

    def run_preliminaries(
        self,
        config,
        cluster_indexed,
        pathway_lookup,
        use_cached=False,
        show_progress=True,
        cached_densityandrate: Optional[Tuple[int, int]] = None,
        name_lookup=None,
    ):
        from timeit import default_timer as timer

        prelim_table = outer_pbar = None
        if show_progress:
            from progress_table import ProgressTable

            prelim_table = ProgressTable(
                default_column_alignment="left", refresh_rate=0
            )
            outer_pbar = prelim_table(
                4,
                description="Preliminary steps",
                show_throughput=False,
                show_progress=True,
                position=2,
            )
            prelim_table.add_column("#", alignment="right", width=3)
            prelim_table.add_column("Step", width=20)
            prelim_table.add_column("Description", width=40)
            prelim_table.add_column("Time (s)", alignment="right")
            prelim_table["#"] = "1/4"
            prelim_table["Step"] = "Skimmer"
            outer_pbar.update()
            start = timer()

            skimmer_np = self._run_skimmer(config)

            prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
            prelim_table.next_row()
        else:
            skimmer_np = self._run_skimmer(config)

        if cached_densityandrate:
            dos_histogram_id, rate_histogram_id = cached_densityandrate
            k_rates, cluster_dos = self.get_cached_densityandrate(
                cluster_indexed, dos_histogram_id, rate_histogram_id, pathway_lookup
            )
        else:
            k_rates, cluster_dos = self.run_densityandrate(
                config,
                cluster_indexed,
                pathway_lookup,
                tablepbar=(prelim_table, outer_pbar) if show_progress else None,
                name_lookup=name_lookup,
            )

        return skimmer_np, k_rates, cluster_dos


class ExperimentRunner:
    """
    This class helps with running the simulation across configs and clusters/pathways in an ExperimentDatabase.

    It also records results and failures back into the database, and can optionally print progress tables to the terminal.

    It precomputes all histograms of density of states and rate constants as needed, and caches them in the database for future runs.
    """

    def __init__(self, db: ExperimentDatabase):
        self.db = db
        self.preparer = DerivedDataPreparer(db)
        self.current_run_id = None

    def _guard_run_started(self):
        if self.current_run_id is None:
            raise RuntimeError("No experiment run started; call start_run() first")

    def start_run(self, config_id=None, **kwargs):
        self.current_run_id = self.db.insert_run(config_id, **kwargs)

    def run_mass_spec(
        self,
        mass_spec,
        subs,
        *args,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
        product_ids=None,
        strict=False,
        strict_dos=True,
        loglevel=0,
        **kwargs,
    ):
        self._guard_run_started()
        if pathway_id is None and cluster_id is None:
            raise ValueError("Either pathway_id or cluster_id must be provided")
        from apitofsim.api import mass_spec as _mass_spec

        event_recorder = None

        if isinstance(self.db, RealizationDatabase):
            event_recorder = EventRecorder(
                self.db, pathway_ids if pathway_ids is not None else [pathway_id]
            )
            logconf = (loglevel, True)
        else:
            logconf = (loglevel, False)

        counters = None
        try:
            counters, timings = _mass_spec(
                mass_spec,
                subs,
                *args,
                **kwargs,
                logconf=logconf,
                event_callback=event_recorder,
                named_tuple_counters=True,
                output_timings=True,
                strict=strict_dos,
            )
        except ApiTofError as e:
            if strict:
                raise
            overflow_requested = None
            if isinstance(e, ApiTofOverflowError):
                overflow_requested = e.current
            experiment_result_id = self.db.record_failure(
                self.current_run_id,
                type(e).__name__,
                str(e),
                overflow_requested,
                pathway_id=pathway_id,
                cluster_id=cluster_id,
            )
        else:
            experiment_result_id = self.db.record_result(
                self.current_run_id,
                counters,
                timings,
                pathway_id=pathway_id,
                pathway_ids=pathway_ids,
                cluster_id=cluster_id,
            )
        if event_recorder is not None:
            event_recorder.relate_realizations(experiment_result_id)
        return counters

    def _prepare_from_config(
        self,
        config,
        parent=None,
        pathways=None,
    ):
        cluster_indexed, name_lookup, pathway_lookup = self.db.get_all_lookups(
            parent, pathways
        )

        skimmer_np, k_rates, cluster_dos = self.preparer.run_preliminaries(
            config,
            cluster_indexed,
            pathway_lookup=pathway_lookup,
            name_lookup=name_lookup,
        )

        assert isinstance(skimmer_np, numpy.ndarray)
        mass_spec = MassSpectrometer(
            skimmer_np,
            config["lengths"],
            config["voltages"],
            config["T"],
            config["pressures"],
            quadrupole=config.get("quadrupole"),
        )

        return (
            mass_spec,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        )

    """
    Run a `config` passed directly as a dict.
    """

    def run_from_config(
        self,
        config,
        run_started=False,
        strict_dos=True,
        mode=SimulationMode.SINGLE_CLUSTER,
        parent=None,
        pathways=None,
        verbose=False,
    ):
        if not run_started:
            self.start_run()

        (
            mass_spec,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        ) = self._prepare_from_config(
            config,
            parent=parent,
            pathways=pathways,
        )

        if mode == SimulationMode.PATHWAY_AT_A_TIME:
            self._run_pathways_at_a_time(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
                verbose=verbose,
            )
        elif mode == SimulationMode.SINGLE_CLUSTER:
            self._run_cluster_grouped(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
                verbose=verbose,
            )
        else:
            assert mode == SimulationMode.CLUSTER_TREE
            self._run_cluster_tree(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
                verbose=verbose,
            )

    def _mk_update_from_counters(self, table, pbar):
        def update_from_counters(counters):
            fragmented_total = counter_fragmented_total(counters)
            realizations = counters.n_realizations
            table["Frags"] = int(fragmented_total)
            table["Intacts"] = int(counters.n_escaped_total)
            table["Avg colls"] = counters.ncoll_total / realizations
            table["PH rej"] = int(counters.counter_collision_rejections)
            table["Surv prob"] = counters.n_escaped_total / realizations
            table["Warns"] = int(counters.nwarnings)
            pbar.set_step(realizations)

        return update_from_counters

    def _run_pathways_at_a_time(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
        parent=None,
        verbose=False,
    ):
        from os import environ
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        from apitofsim.api import Histogram

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            refresh_rate=0,
            interactive=0 if verbose else int(environ.get("PTABLE_INTERACTIVE", "2")),
        )
        mass_spec_table.add_column("#")
        mass_spec_table.add_column("Cluster", alignment="left")
        mass_spec_table.add_column("Products", width=16, alignment="left")
        mass_spec_table.add_columns(
            "Frags",
            "Intacts",
            "Avg colls",
            "PH rej",
        )
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            pathway_lookup.items(),
            description="Pathway",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )
        for pathway_id, (cluster_id, product1_id, product2_id) in outer_pbar:
            density_cluster = cluster_dos[cluster_id]
            rate_const = k_rates[pathway_id]
            cluster = cluster_indexed[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            mass_spec_table["#"] = f"{cluster_seq}/{len(pathway_lookup)}"
            mass_spec_table["Cluster"] = name_lookup[cluster_id]
            mass_spec_table["Products"] = (
                f"{name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
            start = timer()
            density_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max"],
                density_cluster,
            )
            rate_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max_rate"],
                rate_const,
            )
            subs = MassSpecSubstanceSingleInput(
                cluster,
                product1,
                product2,
                config["gas"],
                density_hist,
                rate_hist,
                fragmentation_energy=config.get("fragmentation_energy"),
                cluster_charge_sign=cluster.charge,
            )
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )
            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            def log_callback(typ, msg):
                msg = msg.rstrip()
                print(f"Channel: {typ}; Msg: {msg}")

            update_from_counters = self._mk_update_from_counters(
                mass_spec_table, inner_pbar
            )
            counters = self.run_mass_spec(
                mass_spec,
                subs,
                realizations,
                pathway_id=pathway_id,
                sample_mode=SampleMode.rejection,
                loglevel=1 if verbose else 0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
                result_callback=update_from_counters,
                log_callback=log_callback,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def _run_cluster_grouped(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
        verbose=False,
    ):
        from os import environ
        from timeit import default_timer as timer
        from typing import Any

        from progress_table import ProgressTable

        from apitofsim.api import Histogram

        last_cluster_id = None
        groups = []
        cur_group: dict[str, Any] | None = None
        for pathway_id, (
            cluster_id,
            product1_id,
            product2_id,
        ) in pathway_lookup.items():
            rate_const = k_rates[pathway_id]
            density_cluster = cluster_dos[cluster_id]
            cluster = cluster_indexed[cluster_id]
            if cluster_id != last_cluster_id:
                if cur_group is not None:
                    groups.append(cur_group)
                density_hist = Histogram.from_mesh(
                    config["bin_width"],
                    config["energy_max"],
                    density_cluster,
                )
                cur_group = {
                    "pathways": [],
                    "cluster": cluster,
                    "cluster_id": cluster_id,
                    "cluster_label": name_lookup[cluster_id],
                    "density_hist": density_hist,
                    "product_labels": [],
                    "pathway_ids": [],
                }
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            rate_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max_rate"],
                rate_const,
            )
            assert cur_group is not None
            cur_group["pathways"].append(
                MassSpecInputFragmentationPathway(
                    cluster, product1, product2, rate_hist
                )
            )
            cur_group["product_labels"].append(
                f"{name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
            cur_group["pathway_ids"].append(pathway_id)
            last_cluster_id = cluster_id
        if cur_group is not None:
            groups.append(cur_group)

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            refresh_rate=0,
            interactive=0 if verbose else int(environ.get("PTABLE_INTERACTIVE", "2")),
        )
        mass_spec_table.add_column("Cluster", width=16, alignment="left")
        mass_spec_table.add_column("Paths", width=5, alignment="left")
        mass_spec_table.add_column("Frags", width=5)
        mass_spec_table.add_columns(
            "Intacts",
            "Avg colls",
        )
        mass_spec_table.add_column("PH rej", width=6)
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            groups,
            description="Cluster",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )
        for group in outer_pbar:
            mass_spec_table["Cluster"] = group["cluster_label"]
            mass_spec_table["Paths"] = str(len(group["pathways"]))
            start = timer()
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )
            cluster = group["cluster"]
            subs = MassSpecSubstanceSingleInput(
                cluster,
                group["pathways"],
                config["gas"],
                group["density_hist"],
                cluster.charge,
            )
            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            update_from_counters = self._mk_update_from_counters(
                mass_spec_table, inner_pbar
            )
            counters = self.run_mass_spec(
                mass_spec,
                subs,
                realizations,
                cluster_id=group["cluster_id"],
                pathway_ids=group["pathway_ids"],
                sample_mode=SampleMode.rejection,
                loglevel=1 if verbose else 0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
                result_callback=update_from_counters,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def _prepare_cluster_tree_lookup(
        self,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
    ):
        from apitofsim.api import Histogram

        cluster_lookup = {}
        for cluster_id, cluster in cluster_indexed.items():
            density_cluster = cluster_dos[cluster_id]
            density_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max"],
                density_cluster,
            )
            cluster_lookup[cluster_id] = {
                "pathways": [],
                "cluster": cluster,
                "cluster_id": cluster_id,
                "cluster_label": name_lookup[cluster_id],
                "density_hist": density_hist,
                "product_labels": [],
                "pathway_products": [],
                "pathway_ids": [],
            }

        for pathway_id, (
            cluster_id,
            product1_id,
            product2_id,
        ) in pathway_lookup.items():
            rate_const = k_rates[pathway_id]
            cluster = cluster_indexed[cluster_id]
            cur_cluster_dict = cluster_lookup[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            rate_hist = Histogram.from_mesh(
                config["bin_width"],
                config["energy_max_rate"],
                rate_const,
            )
            cur_cluster_dict["pathways"].append(
                MassSpecInputFragmentationPathway(
                    cluster, product1, product2, rate_hist
                )
            )
            cur_cluster_dict["product_labels"].append(
                f"{name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
            cur_cluster_dict["pathway_products"].append((product1_id, product2_id))
            cur_cluster_dict["pathway_ids"].append(pathway_id)
        return cluster_lookup

    def _prepare_cluster_tree(
        self,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
    ):
        cluster_lookup = self._prepare_cluster_tree_lookup(
            config,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        )

        viable_roots = [
            cluster
            for cluster in cluster_lookup.values()
            if cluster["pathway_products"]
        ]

        def build_tree(root, parents=()):
            cluster_id = root["cluster_id"]
            if cluster_id in parents:
                raise ValueError(
                    f"Cycle detected in cluster tree with parents {parents} and current cluster {cluster_id}"
                )
            children = []
            new_parents = (*parents, cluster_id)
            for pathway, products in zip(root["pathways"], root["pathway_products"]):
                children.append(
                    (
                        pathway,
                        build_tree(cluster_lookup[products[0]], parents=new_parents),
                        build_tree(cluster_lookup[products[1]], parents=new_parents),
                    )
                )
            return (root["cluster"], root["density_hist"], children)

        for root in viable_roots:
            pathway_ids = []
            product_ids = []

            def mk_pathway_ids(root, parents=()):
                new_parents = (*parents, root["cluster_id"])
                for (product1_id, product2_id), pathway_id in zip(
                    root["pathway_products"], root["pathway_ids"]
                ):
                    pathway_ids.append(pathway_id)
                    for product_id in (product1_id, product2_id):
                        if product_id in parents:
                            raise ValueError(
                                f"Cycle detected in cluster tree with parents {parents} and current product {product_id}"
                            )
                        product_ids.append(product_id)
                        mk_pathway_ids(cluster_lookup[product_id], new_parents)

            mk_pathway_ids(root)
            tree = build_tree(root)

            yield (pathway_ids, product_ids, tree, root)

    def _run_cluster_tree(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
        verbose=False,
    ):
        from os import environ
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            refresh_rate=0,
            interactive=0 if verbose else int(environ.get("PTABLE_INTERACTIVE", "2")),
        )
        mass_spec_table.add_column("Cluster", width=16, alignment="left")
        mass_spec_table.add_column("Paths", width=5, alignment="left")
        mass_spec_table.add_column("Frags", width=5)
        mass_spec_table.add_columns(
            "Intacts",
            "Avg colls",
        )
        mass_spec_table.add_column("PH rej", width=6)
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            self._prepare_cluster_tree(
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
            ),
            description="Cluster",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )

        for pathway_ids, product_ids, tree, root in outer_pbar:
            mass_spec_table["Cluster"] = root["cluster_label"]
            mass_spec_table["Paths"] = str(len(root["pathways"]))
            start = timer()
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )
            subs = MassSpecSubstanceTreeInput(config["gas"], tree)
            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            update_from_counters = self._mk_update_from_counters(
                mass_spec_table, inner_pbar
            )
            counters = self.run_mass_spec(
                mass_spec,
                subs,
                realizations,
                cluster_id=root["cluster_id"],
                pathway_ids=pathway_ids,
                sample_mode=SampleMode.rejection,
                loglevel=1 if verbose else 0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
                result_callback=update_from_counters,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def run_prepared_config(self, name=None, **kwargs):
        """
        Run from an experiment config that has been inserted into an ExperimentDatabase.

         * `name` is the name of an experiment config, a list thereof, or None to run all configs
        """
        from pprint import pprint

        configs = list(self.db.iter_configs(name))
        for idx, row in enumerate(configs):
            print(f"# Running experiment config: {row.name} [{idx + 1}/{len(configs)}]")
            pprint(row.config)
            print()
            self.start_run(
                row.id,
                simulation_mode=kwargs.get(
                    "simulation_mode", SimulationMode.SINGLE_CLUSTER
                ),
            )
            self.run_from_config(row.config, run_started=True, **kwargs)
            print()
