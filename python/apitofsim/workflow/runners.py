from typing import Callable

import duckdb
import numpy
from pint import get_application_registry

from apitofsim.api import (
    ApiTofError,
    ApiTofOverflowError,
    MassSpecInputFragmentationPathway,
    MassSpecSubstanceInput,
    MassSpectrometer,
    MeshMode,
    defaults,
)

from .db import ExperimentDatabase, SuperClusterDatabase
from .db_utils import get_or_insert, get_through_join_else, insert_via_arrow

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
            P0=config["pressure_first"],
            rmax=config["lengths"][-1],
            dc=config["dc"],
            alpha_factor=config["alpha_factor"],
            gas=config["gas"],
            N=config["N_iter"],
            M=config["M_iter"],
            resolution=config["resolution"],
            tolerance=config["tolerance"],
        )

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

        from apitofsim.api import ProductsCluster, compute_density_of_states_batch

        num_clusters_missed = 0
        cluster_dos_dict = {}
        missed_cluster_ids = []
        wanted_cluster_ids = numpy.array(list(cluster_indexed.keys()))
        density_of_states_inputs = []
        for miss_info in get_through_join_else(
            self.db.db,
            self.db.db.table("cluster_dos"),
            "data",
            cluster_dos_dict,
            cluster_id=wanted_cluster_ids,
            histogram_params_id=histogram_id,
        ):
            cluster_id = miss_info["cluster_id"]
            cluster = cluster_indexed[cluster_id]
            if cluster.is_atom_like_product():
                cluster_dos_dict[cluster_id] = None
            else:
                missed_cluster_ids.append(cluster_id)
                density_of_states_inputs.append(cluster)
                num_clusters_missed += 1

        wanted_p1 = []
        wanted_p2 = []
        for _, product1_id, product2_id in pathway_lookup.values():
            product1_id, product2_id = sorted((product1_id, product2_id))
            wanted_p1.append(product1_id)
            wanted_p2.append(product2_id)

        for miss_info in get_through_join_else(
            self.db.db,
            self.db.db.table("products_dos"),
            "data",
            cluster_dos_dict,
            cluster1_id=wanted_p1,
            cluster2_id=wanted_p2,
            histogram_params_id=histogram_id,
        ):
            p1 = miss_info["cluster1_id"]
            p2 = miss_info["cluster2_id"]
            products = ProductsCluster(cluster_indexed[p1], cluster_indexed[p2])
            density_of_states_inputs.append(products)

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
        histogram_id,
        mesh,
        bin_width,
        energy_max,
        energy_max_rate,
        pathway_lookup,
        status_table=None,
    ):
        import pyarrow as pa

        from apitofsim import (
            FragmentationPathway,
            KTotalInput,
            compute_k_total_batch,
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

        for miss_info in get_through_join_else(
            self.db.db,
            self.db.db.table("k_rate").filter(
                duckdb.ColumnExpression("histogram_params_id")
                == duckdb.ConstantExpression(histogram_id)
            ),
            "data",
            k_total_dict,
            pathway_id=pathway_lookup.keys(),
        ):
            pathway_id = miss_info["pathway_id"]
            cluster_id, product1_id, product2_id = pathway_lookup[pathway_id]
            product1_cpp = cluster_indexed[product1_id].into_cpp()
            product2_cpp = cluster_indexed[product2_id].into_cpp()
            fragmentation_energy = FragmentationPathway(
                cluster_indexed[cluster_id].into_cpp(), product1_cpp, product2_cpp
            ).fragmentation_energy_kelvin()
            validate_max_energies(
                fragmentation_energy=fragmentation_energy,
                bin_width=bin_width,
                energy_max=energy_max,
                energy_max_rate=energy_max_rate,
                quantities_strict=False,
            )
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
        self, config, cluster_indexed, pathway_lookup, tablepbar=None
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
            histogram_id,
            mesh,
            config["bin_width"],
            config["energy_max"],
            config["energy_max_rate"],
            pathway_lookup=pathway_lookup,
            status_table=table,
        )

        table["Time (s)"] = f"{(timer() - start):.2f}"
        table.close()
        if pbar._is_active:
            # Not sure why this is sometimes needed
            pbar.close()

        cluster_dos_by_pathway = {}

        for pathway_id, (
            _,
            product1_id,
            product2_id,
        ) in pathway_lookup.items():
            product1_id, product2_id = sorted((product1_id, product2_id))
            cluster_dos_by_pathway[pathway_id] = cluster_dos_dict[
                (product1_id, product2_id)
            ]

        return k_rates, cluster_dos_by_pathway

    def run_preliminaries(self, config, cluster_indexed, pathway_lookup):
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        prelim_table = ProgressTable(default_column_alignment="left", refresh_rate=0)
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

        k_rates, cluster_dos_by_pathway = self.run_densityandrate(
            config,
            cluster_indexed,
            pathway_lookup,
            tablepbar=(prelim_table, outer_pbar),
        )

        return skimmer_np, k_rates, cluster_dos_by_pathway


class ExperimentRunner:
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
        *args,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
        strict=False,
        strict_dos=True,
        **kwargs,
    ):
        self._guard_run_started()
        if pathway_id is None and cluster_id is None:
            raise ValueError("Either pathway_id or cluster_id must be provided")
        from apitofsim.api import mass_spec

        counters = None
        try:
            counters, timings = mass_spec(
                *args,
                **kwargs,
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
            self.db.record_failure(
                self.current_run_id,
                type(e).__name__,
                str(e),
                overflow_requested,
                pathway_id=pathway_id,
                cluster_id=cluster_id,
            )
        else:
            self.db.record_result(
                self.current_run_id,
                counters,
                timings,
                pathway_id=pathway_id,
                pathway_ids=pathway_ids,
                cluster_id=cluster_id,
            )
        return counters

    def run_from_config(
        self,
        config,
        run_started=False,
        strict_dos=True,
        pathway_at_a_time=False,
        parent=None,
        pathways=None,
        verbose=False,
    ):
        if not run_started:
            self.start_run()

        cluster_indexed, name_lookup, pathway_lookup = self.db.get_all_lookups(
            parent, pathways
        )

        skimmer_np, k_rates, cluster_dos = self.preparer.run_preliminaries(
            config, cluster_indexed, pathway_lookup=pathway_lookup
        )

        assert isinstance(skimmer_np, numpy.ndarray)
        mass_spec = MassSpectrometer(
            skimmer_np,
            config["lengths"],
            config["voltages"],
            config["T"],
            Q_(
                numpy.array(
                    [
                        config["pressure_first"].to("pascals").magnitude,
                        config["pressure_second"].to("pascals").magnitude,
                    ]
                ),
                "pascals",
            ),
            quadrupole=config.get("quadrupole"),
        )

        if pathway_at_a_time:
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
        else:
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

    def _mk_update_from_counters(self, table, pbar):
        def update_from_counters(counters):
            fragmented_total = counter_fragmented_total(counters)
            realizations = fragmented_total + counters.n_escaped_total
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
            density_cluster = cluster_dos[pathway_id]
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
            subs = MassSpecSubstanceInput(
                cluster,
                product1,
                product2,
                config["gas"],
                density_hist,
                rate_hist,
                fragmentation_energy=config.get("fragmentation_energy"),
                cluster_charge_sign=config.get(
                    "cluster_charge_sign", defaults.cluster_charge_sign
                ),
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
                sample_mode=2,
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
            density_cluster = cluster_dos[pathway_id]
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
            subs = MassSpecSubstanceInput(
                group["cluster"],
                group["pathways"],
                config["gas"],
                group["density_hist"],
                config.get("cluster_charge_sign", defaults.cluster_charge_sign),
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
                sample_mode=2,
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
        from pprint import pprint

        configs = list(self.db.iter_configs(name))
        for idx, row in enumerate(configs):
            print(f"# Running experiment config: {row.name} [{idx + 1}/{len(configs)}]")
            pprint(row.config)
            print()
            self.start_run(
                row.id, pathway_at_a_time=kwargs.get("pathway_at_a_time", False)
            )
            self.run_from_config(row.config, run_started=True, **kwargs)
            print()
