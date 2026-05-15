"""
Connectivity CSV + Arbor Recipe Generation
Neural Reconstruction Pipeline - Phase 4

Consumes outputs from all previous phases and produces two deliverables:

1. connectivity.csv
   One row per confirmed synapse (SYNAPSE segments) and per potential
   contact (CONTACT segments).  Columns give both the presynaptic (axon)
   and postsynaptic (POST_SYN) cable locations in Arbor-addressable form:

       pre_cell,  pre_swc,  pre_edge_u,  pre_edge_v,  pre_arc_frac,
       post_cell, post_swc, post_edge_u, post_edge_v, post_arc_frac,
       synapse_name, synapse_seg_id,
       bouton_name,  bouton_seg_id,
       distance_pre_um, distance_post_um,
       connection_type   (synapse | contact)

2. arbor_recipe.json
   A minimal Arbor network description that Arbor's Python API can ingest:
   - cell_labels   - maps each cell name to its SWC file
   - synapses      - list of (pre_loc, post_loc, mechanism) dicts
   - contacts      - list of (pre_loc, contact_loc) dicts  (no mechanism)

   Locations follow the Arbor cable-cell location format:
       {"cell": "A1", "branch": <swc_row_id>, "pos": <arc_frac>}

   branch is the SWC row ID (pre_swc_u / post_swc_u from connectivity.csv).
   Entries where those IDs are absent are omitted from the recipe and logged.

"""

import csv
import json
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from neuron_pipeline.stages.segment_classifier import SegmentRegistry
from neuron_pipeline.stages.centroid_mapper import CableMappingTable, CableMappingEntry


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class SynapseRow:
    """One confirmed synapse or putative contact connection."""
    connection_type: str            # 'synapse' | 'contact'

    # Presynaptic (axon) side
    pre_cell: str
    pre_swc: str
    pre_edge_u: int
    pre_edge_v: int
    pre_arc_frac: float
    pre_nearest_x_um: float
    pre_nearest_y_um: float
    pre_nearest_z_um: float
    distance_pre_um: float

    # Postsynaptic side (POST_SYN cell) -- empty string / -1 for contacts
    post_cell: str
    post_swc: str
    post_edge_u: int
    post_edge_v: int
    post_arc_frac: float
    post_nearest_x_um: float
    post_nearest_y_um: float
    post_nearest_z_um: float
    distance_post_um: float

    # Annotation identifiers
    synapse_name: str
    synapse_seg_id: int
    bouton_name: str
    bouton_seg_id: int

    # QC and SWC row IDs (from Phase 3 CableMappingEntry)
    pre_qc_flag:  str           = 'ok'
    pre_swc_u:    Optional[int] = None   # SWC row ID of pre_edge_u node
    pre_swc_v:    Optional[int] = None   # SWC row ID of pre_edge_v node
    post_qc_flag: str           = 'ok'
    post_swc_u:   Optional[int] = None   # SWC row ID of post_edge_u node
    post_swc_v:   Optional[int] = None   # SWC row ID of post_edge_v node


@dataclass
class ConnectivityOutput:
    rows: List[SynapseRow] = field(default_factory=list)

    def add(self, row: SynapseRow) -> None:
        self.rows.append(row)

    def __len__(self) -> int:
        return len(self.rows)

    def write_csv(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self.rows:
            Path(path).write_text("")
            return path
        fieldnames = list(asdict(self.rows[0]).keys())
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                d = asdict(row)
                for k in ('pre_arc_frac', 'post_arc_frac'):
                    d[k] = f"{d[k]:.6f}"
                for k in ('distance_pre_um', 'distance_post_um',
                          'pre_nearest_x_um', 'pre_nearest_y_um', 'pre_nearest_z_um',
                          'post_nearest_x_um', 'post_nearest_y_um', 'post_nearest_z_um'):
                    d[k] = f"{d[k]:.4f}"
                writer.writerow(d)
        return path


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

# Default excitatory synapse mechanism name (override as needed)
_DEFAULT_SYN_MECHANISM = "expsyn"


class ConnectivityBuilder:
    """
    Build connectivity.csv and arbor_recipe.json from pipeline outputs.

    Synapse resolution
    ------------------
    Each SYNAPSE segment (A1B2P1S1) defines a confirmed connection:
      - presynaptic  location -> cable mapping of the parent BOUTON  (A1B2)
      - postsynaptic location -> cable mapping of the parent POST_SYN (A1B2P1)

    Contact resolution
    ------------------
    Each CONTACT segment (A1B2X1) defines a potential (unconfirmed) contact:
      - presynaptic  location -> cable mapping of the parent BOUTON  (A1B2)
      - postsynaptic location -> not available (no POST_SYN defined)
        -> post_* fields are left as empty / -1

    Both are written to connectivity.csv; contacts are flagged in the
    connection_type column and omitted from the Arbor synapses list
    (they appear in a separate contacts list in the recipe).
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build(
        self,
        registry: SegmentRegistry,
        trees: Dict[str, Tuple],           # cell_name -> (tree, swc_path)
        mapping_table: CableMappingTable,
        output_dir: str = "./vast_export",
        syn_mechanism: str = _DEFAULT_SYN_MECHANISM,
        cell_mechanisms: Optional[Dict[str, List[str]]] = None,
    ) -> Tuple[ConnectivityOutput, str, str]:
        """
        Build connectivity table and Arbor recipe.

        Args:
            registry      : SegmentRegistry from Phase 0.
            trees         : Phase 1 output -- cell_name -> (tree, swc_path).
            mapping_table : Phase 3 output -- CableMappingTable.
            output_dir    : Base directory; files written here.
            syn_mechanism : Arbor mechanism name for confirmed synapses.

        Returns:
            (connectivity, csv_path, recipe_path)
        """
        # Index mapping table by centroid name for O(1) lookups
        by_name: Dict[str, CableMappingEntry] = {
            e.centroid_name: e for e in mapping_table.entries
        }

        # SWC paths indexed by cell name
        swc_paths: Dict[str, str] = {
            name: swc_path for name, (_, swc_path) in trees.items()
        }

        connectivity = ConnectivityOutput()

        # ---- Confirmed synapses ----
        n_syn_ok = n_syn_skip = 0
        for row in registry.connectivity:
            result = self._build_synapse_row(row, by_name, swc_paths, registry)
            if result is not None:
                connectivity.add(result)
                n_syn_ok += 1
            else:
                n_syn_skip += 1

        # ---- Contacts ----
        n_con_ok = n_con_skip = 0
        for row in registry.contacts:
            result = self._build_contact_row(row, by_name, swc_paths)
            if result is not None:
                connectivity.add(result)
                n_con_ok += 1
            else:
                n_con_skip += 1

        self.logger.info(
            f"Connectivity: {n_syn_ok} synapses ({n_syn_skip} skipped), "
            f"{n_con_ok} contacts ({n_con_skip} skipped)"
        )

        # Write CSV
        csv_path = str(Path(output_dir) / "connectivity.csv")
        connectivity.write_csv(csv_path)
        self.logger.info(f"Written: {csv_path}")

        # Write Arbor recipe JSON
        recipe_path = str(Path(output_dir) / "arbor_recipe.json")
        self._write_arbor_recipe(
            connectivity, swc_paths, registry, recipe_path, syn_mechanism, cell_mechanisms
        )
        self.logger.info(f"Written: {recipe_path}")

        return connectivity, csv_path, recipe_path

    # ------------------------------------------------------------------
    # Row builders
    # ------------------------------------------------------------------

    def _build_synapse_row(
        self,
        conn_row,                           # ConnectivityRow namedtuple
        by_name: Dict[str, CableMappingEntry],
        swc_paths: Dict[str, str],
        registry: SegmentRegistry,
    ) -> Optional[SynapseRow]:
        """
        Build one SynapseRow for a confirmed synapse.

        Presynaptic  = BOUTON cable location.
        Postsynaptic = POST_SYN cable location.
        """
        bouton_map   = by_name.get(conn_row.bouton_name)
        post_syn_map = by_name.get(conn_row.post_syn_name)

        if bouton_map is None:
            self.logger.debug(
                f"  SYNAPSE {conn_row.synapse_name}: no cable mapping for "
                f"BOUTON {conn_row.bouton_name} -- skipped"
            )
            return None

        # Postsynaptic side: the POST_SYN cell is its own skeleton.
        # Its centroid may or may not have been mapped (Phase 3 only maps
        # BOUTON/SYNAPSE/CONTACT, not POST_SYN, which has a full SWC).
        # Use the post_syn cable mapping if available, otherwise leave blank.
        if post_syn_map is not None:
            post_cell      = post_syn_map.cell_name
            post_swc       = swc_paths.get(post_cell, "")
            post_edge_u    = post_syn_map.edge_u
            post_edge_v    = post_syn_map.edge_v
            post_arc_frac  = post_syn_map.arc_fraction
            post_nx        = post_syn_map.nearest_x_um
            post_ny        = post_syn_map.nearest_y_um
            post_nz        = post_syn_map.nearest_z_um
            dist_post      = post_syn_map.distance_um
        else:
            # Fall back: look for the POST_SYN cell directly in trees
            post_cell      = conn_row.post_syn_name
            post_swc       = swc_paths.get(conn_row.post_syn_name, "")
            post_edge_u    = -1
            post_edge_v    = -1
            post_arc_frac  = 0.0
            post_nx = post_ny = post_nz = 0.0
            dist_post      = -1.0
            self.logger.debug(
                f"  SYNAPSE {conn_row.synapse_name}: no cable mapping for "
                f"POST_SYN {conn_row.post_syn_name} -- post side left blank"
            )

        return SynapseRow(
            connection_type="synapse",
            pre_cell=bouton_map.cell_name,
            pre_swc=swc_paths.get(bouton_map.cell_name, ""),
            pre_edge_u=bouton_map.edge_u,
            pre_edge_v=bouton_map.edge_v,
            pre_arc_frac=bouton_map.arc_fraction,
            pre_nearest_x_um=bouton_map.nearest_x_um,
            pre_nearest_y_um=bouton_map.nearest_y_um,
            pre_nearest_z_um=bouton_map.nearest_z_um,
            distance_pre_um=bouton_map.distance_um,
            post_cell=post_cell,
            post_swc=post_swc,
            post_edge_u=post_edge_u,
            post_edge_v=post_edge_v,
            post_arc_frac=post_arc_frac,
            post_nearest_x_um=post_nx,
            post_nearest_y_um=post_ny,
            post_nearest_z_um=post_nz,
            distance_post_um=dist_post,
            synapse_name=conn_row.synapse_name,
            synapse_seg_id=conn_row.synapse_seg_id,
            bouton_name=conn_row.bouton_name,
            bouton_seg_id=conn_row.bouton_seg_id,
            pre_qc_flag=getattr(bouton_map,   'qc_distance_flag', 'ok'),
            pre_swc_u=getattr(bouton_map,     'swc_node_u', None),
            pre_swc_v=getattr(bouton_map,     'swc_node_v', None),
            post_qc_flag=getattr(post_syn_map, 'qc_distance_flag', 'ok') if post_syn_map else 'no_skeleton',
            post_swc_u=getattr(post_syn_map,   'swc_node_u', None)       if post_syn_map else None,
            post_swc_v=getattr(post_syn_map,   'swc_node_v', None)       if post_syn_map else None,
        )

    def _build_contact_row(
        self,
        contact_row,                        # ContactRow namedtuple
        by_name: Dict[str, CableMappingEntry],
        swc_paths: Dict[str, str],
    ) -> Optional[SynapseRow]:
        """
        Build one SynapseRow for a putative contact (no POST_SYN partner).
        """
        bouton_map = by_name.get(contact_row.bouton_name)
        if bouton_map is None:
            self.logger.debug(
                f"  CONTACT {contact_row.contact_name}: no cable mapping for "
                f"BOUTON {contact_row.bouton_name} -- skipped"
            )
            return None

        return SynapseRow(
            connection_type="contact",
            pre_cell=bouton_map.cell_name,
            pre_swc=swc_paths.get(bouton_map.cell_name, ""),
            pre_edge_u=bouton_map.edge_u,
            pre_edge_v=bouton_map.edge_v,
            pre_arc_frac=bouton_map.arc_fraction,
            pre_nearest_x_um=bouton_map.nearest_x_um,
            pre_nearest_y_um=bouton_map.nearest_y_um,
            pre_nearest_z_um=bouton_map.nearest_z_um,
            distance_pre_um=bouton_map.distance_um,
            post_cell="",
            post_swc="",
            post_edge_u=-1,
            post_edge_v=-1,
            post_arc_frac=0.0,
            post_nearest_x_um=0.0,
            post_nearest_y_um=0.0,
            post_nearest_z_um=0.0,
            distance_post_um=-1.0,
            synapse_name=contact_row.contact_name,
            synapse_seg_id=contact_row.contact_seg_id,
            bouton_name=contact_row.bouton_name,
            bouton_seg_id=contact_row.bouton_seg_id,
            pre_qc_flag=getattr(bouton_map, 'qc_distance_flag', 'ok'),
            pre_swc_u=getattr(bouton_map,   'swc_node_u', None),
            pre_swc_v=getattr(bouton_map,   'swc_node_v', None),
        )

    # ------------------------------------------------------------------
    # Arbor recipe
    # ------------------------------------------------------------------

    def _write_arbor_recipe(
        self,
        connectivity: ConnectivityOutput,
        swc_paths: Dict[str, str],
        registry: SegmentRegistry,
        path: str,
        syn_mechanism: str,
        cell_mechanisms: Optional[Dict[str, List[str]]] = None,
    ) -> None:
        """
        Write arbor_recipe.json.

        Schema
        ------
        {
          "metadata": {
            "coord_frame": "physical_um_xyz_center",
            "branch_id_convention": "swc_row_id"
          },
          "cell_labels": {"A1": "swc/cell_A1.swc", ...},
          "cell_mechanisms": {"A1": ["hh", "pas"], ...},   // only if provided
          "synapses": [
            {
              "pre":  {"cell": "A1", "branch": <swc_row_id>, "pos": <arc_frac>},
              "post": {"cell": "A1B1P1", "branch": <swc_row_id>, "pos": <arc_frac>},
              "mechanism": "expsyn",
              "synapse_name": "A1B1P1S1",
              "bouton_name":  "A1B1"
            },
            ...
          ],
          "contacts": [
            {
              "pre":  {"cell": "A1", "branch": <swc_row_id>, "pos": <arc_frac>},
              "contact_name": "A1B1X1",
              "bouton_name":  "A1B1"
            },
            ...
          ]
        }

        branch is the SWC row ID from pre_swc_u / post_swc_u in connectivity.csv.
        Rows where those IDs are None or -1 are omitted and logged as warnings.
        """
        # Cell labels: all cells with a known SWC path
        cell_labels = {
            name: str(Path(swc_path).name)
            for name, swc_path in swc_paths.items()
        }

        synapses = []
        contacts = []
        n_pre_unmappable = 0
        n_post_unmappable = 0

        for row in connectivity.rows:
            if row.pre_swc_u is None or row.pre_swc_u == -1:
                self.logger.warning(
                    f"  Recipe: {row.synapse_name} pre side has no SWC node "
                    f"(pre_edge_u={row.pre_edge_u}) -- omitted from recipe"
                )
                n_pre_unmappable += 1
                continue

            pre_loc = {
                "cell":   row.pre_cell,
                "branch": row.pre_swc_u,
                "pos":    round(row.pre_arc_frac, 6),
            }

            if row.connection_type == "synapse":
                if row.post_swc_u is None or row.post_swc_u == -1:
                    self.logger.warning(
                        f"  Recipe: {row.synapse_name} post side has no SWC node "
                        f"(post_edge_u={row.post_edge_u}) -- omitted from recipe"
                    )
                    n_post_unmappable += 1
                    continue
                post_loc = {
                    "cell":   row.post_cell,
                    "branch": row.post_swc_u,
                    "pos":    round(row.post_arc_frac, 6),
                }
                synapses.append({
                    "pre":          pre_loc,
                    "post":         post_loc,
                    "mechanism":    syn_mechanism,
                    "synapse_name": row.synapse_name,
                    "bouton_name":  row.bouton_name,
                })

            elif row.connection_type == "contact":
                contacts.append({
                    "pre":          pre_loc,
                    "contact_name": row.synapse_name,
                    "bouton_name":  row.bouton_name,
                })

        if n_pre_unmappable or n_post_unmappable:
            self.logger.warning(
                f"Recipe: {n_pre_unmappable} entries omitted (pre SWC node missing), "
                f"{n_post_unmappable} entries omitted (post SWC node missing)"
            )

        recipe = {
            "metadata": {
                "coord_frame": "physical_um_xyz_center",
                "branch_id_convention": "swc_row_id",
            },
            "cell_labels": cell_labels,
            "synapses":    synapses,
            "contacts":    contacts,
        }
        if cell_mechanisms:
            recipe["cell_mechanisms"] = cell_mechanisms

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(recipe, f, indent=2)

        syn_rows = [r for r in connectivity.rows if r.connection_type == "synapse"]
        if syn_rows:
            n_post_complete = sum(
                1 for r in syn_rows
                if r.post_swc_u is not None and r.post_swc_u != -1
            )
            self.logger.info(
                f"Post-side SWC coverage: {n_post_complete}/{len(syn_rows)} synapses "
                f"have a mapped post_swc_u "
                f"({100 * n_post_complete // len(syn_rows)}%)"
            )

        # Write companion Python recipe class for Arbor users
        py_path = Path(path).with_suffix('.py')
        _template = Path(__file__).parent.parent / "templates" / "arbor_recipe_template.py"
        py_path.write_text(_template.read_text())
        self.logger.info(f"Written: {py_path}")

        self.logger.info(
            f"Arbor recipe: {len(cell_labels)} cells, "
            f"{len(synapses)} synapses, {len(contacts)} contacts"
        )
    
    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def write_analysis(
        self,
        connectivity: ConnectivityOutput,
        registry: SegmentRegistry,
        trees: Dict[str, Tuple],
        output_dir: str = "./vast_export",
    ) -> Tuple[str, str]:
        """
        Write connectivity_summary.csv and connectivity_report.txt.

        Summary CSV: one row per (axon, post_syn) pair -- synapse counts,
        distance stats, and QC flag tallies.

        Report: plain-text with segment inventory, per-pair breakdown,
        missing connections, and QC warnings.

        Returns (summary_csv_path, report_txt_path).
        """
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        summary_path = str(out / "connectivity_summary.csv")
        report_path  = str(out / "connectivity_report.txt")

        syn_rows = [r for r in connectivity.rows if r.connection_type == "synapse"]
        con_rows = [r for r in connectivity.rows if r.connection_type == "contact"]

        # Group synapses by (pre_cell, post_cell)
        pairs: Dict[Tuple[str, str], List[SynapseRow]] = defaultdict(list)
        for row in syn_rows:
            pairs[(row.pre_cell, row.post_cell)].append(row)

        # --- summary CSV ---
        summary = []
        for (axon, post_syn), rows in sorted(pairs.items()):
            pre_dists  = [r.distance_pre_um for r in rows]
            post_dists = [r.distance_post_um for r in rows if r.distance_post_um >= 0]
            qc_pre  = {k: sum(1 for r in rows if r.pre_qc_flag  == k)
                       for k in ('ok', 'warn', 'suspicious')}
            qc_post = {k: sum(1 for r in rows if r.post_qc_flag == k)
                       for k in ('ok', 'warn', 'suspicious')}
            summary.append({
                'axon':               axon,
                'post_syn':           post_syn,
                'n_synapses':         len(rows),
                'bouton_names':       ';'.join(sorted({r.bouton_name for r in rows})),
                'synapse_names':      ';'.join(sorted(r.synapse_name for r in rows)),
                'pre_dist_mean_um':   f"{statistics.mean(pre_dists):.3f}",
                'pre_dist_min_um':    f"{min(pre_dists):.3f}",
                'pre_dist_max_um':    f"{max(pre_dists):.3f}",
                'post_dist_mean_um':  f"{statistics.mean(post_dists):.3f}" if post_dists else 'n/a',
                'post_dist_min_um':   f"{min(post_dists):.3f}"              if post_dists else 'n/a',
                'post_dist_max_um':   f"{max(post_dists):.3f}"              if post_dists else 'n/a',
                'pre_qc_ok':          qc_pre['ok'],
                'pre_qc_warn':        qc_pre['warn'],
                'pre_qc_suspicious':  qc_pre['suspicious'],
                'post_qc_ok':         qc_post['ok'],
                'post_qc_warn':       qc_post['warn'],
                'post_qc_suspicious': qc_post['suspicious'],
            })

        if summary:
            with open(summary_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
                writer.writeheader()
                writer.writerows(summary)
        else:
            Path(summary_path).write_text("")

        # --- detect missing connections ---
        mapped_syn  = {r.synapse_name for r in syn_rows}
        mapped_con  = {r.synapse_name for r in con_rows}   # contact_name stored here
        missing_syn = [r for r in registry.connectivity if r.synapse_name  not in mapped_syn]
        missing_con = [r for r in registry.contacts     if r.contact_name  not in mapped_con]

        # --- text report ---
        W = 60
        L: List[str] = []
        L.append("=" * W)
        L.append("CONNECTIVITY ANALYSIS REPORT")
        L.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L.append("=" * W)

        def section(title: str) -> None:
            L.extend(["", title, "-" * W])

        section("SEGMENT INVENTORY")
        role_counts: Dict[str, int] = defaultdict(int)
        for info in registry.segments.values():
            role_counts[info.role] += 1
        for role in ('AXON', 'BOUTON', 'POST_SYN', 'SYNAPSE', 'CONTACT'):
            L.append(f"  {role:<12}: {role_counts.get(role, 0)}")
        L.append(f"  {'Trees OK':<12}: {len(trees)}")

        section("SYNAPSES")
        L.append(f"  Confirmed synapses : {len(syn_rows)}")
        L.append(f"  Axon->PostSyn pairs: {len(pairs)}")
        for (axon, post_syn), rows in sorted(pairs.items()):
            n_susp = sum(1 for r in rows if r.pre_qc_flag == 'suspicious')
            qc_tag = f"  [{n_susp}/{len(rows)} pre suspicious]" if n_susp else ""
            L.append(
                f"\n  {axon} -> {post_syn}  "
                f"({len(rows)} synapse{'s' if len(rows) > 1 else ''}){qc_tag}"
            )
            for r in sorted(rows, key=lambda x: x.synapse_name):
                L.append(
                    f"    {r.synapse_name:<18}"
                    f"  pre {r.distance_pre_um:.2f} um [{r.pre_qc_flag}]"
                    f"  post {'no skeleton' if r.distance_post_um < 0 else f'{r.distance_post_um:.2f} um'} [{r.post_qc_flag}]"
                )

        section("CONTACTS (unconfirmed)")
        L.append(f"  Putative contacts: {len(con_rows)}")
        for r in con_rows:
            L.append(
                f"  {r.synapse_name}  axon={r.pre_cell}  bouton={r.bouton_name}"
                f"  pre {r.distance_pre_um:.2f} um [{r.pre_qc_flag}]"
            )

        section("MISSING CONNECTIONS")
        if not missing_syn and not missing_con:
            L.append("  All expected connections resolved successfully.")
        for row in missing_syn:
            L.append(f"  SYNAPSE {row.synapse_name}  ({row.axon_name} -> {row.post_syn_name})")
            if row.axon_name    not in trees:
                L.append(f"    axon skeleton '{row.axon_name}' failed in Phase 1")
            if row.post_syn_name not in trees:
                L.append(f"    post_syn skeleton '{row.post_syn_name}' failed in Phase 1")
        for row in missing_con:
            L.append(f"  CONTACT {row.contact_name}  (axon={row.axon_name})")
            if row.axon_name not in trees:
                L.append(f"    axon skeleton '{row.axon_name}' failed in Phase 1")

        section("QC FLAGS")
        for tier in ('ok', 'warn', 'suspicious'):
            n_pre  = sum(1 for r in connectivity.rows if r.pre_qc_flag  == tier)
            n_post = sum(1 for r in syn_rows          if r.post_qc_flag == tier)
            L.append(f"  {tier:<12}: pre={n_pre}, post={n_post}")
        if any(r.pre_qc_flag == 'suspicious' for r in connectivity.rows):
            L.append(
                "\n  NOTE: suspicious pre-distances (>5 um) indicate the skeleton\n"
                "  does not reach the bouton location. Check bbox_min_vox fix and\n"
                "  re-run Phase 1 before trusting these numbers."
            )

        L.extend(["", "=" * W, ""])

        Path(report_path).write_text('\n'.join(L))

        # self.logger.info(f"Written: {summary_path}")
        # self.logger.info(f"Written: {report_path}")
        return summary_path, report_path