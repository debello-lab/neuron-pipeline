"""
Connectivity CSV + Arbor Recipe Generation
Neural Reconstruction Pipeline - Phase 4

Consumes outputs from all previous phases and produces two deliverables:

1. connectivity.csv
   One row per confirmed synapse (SYNAPSE segments) and per potential
   contact (CONTACT segments).  Columns give both the presynaptic (axon)
   and postsynaptic (POST_SYN) cable locations in Arbor-addressable form:

       pre_cell, pre_swc, pre_edge_u, pre_edge_v, pre_arc_frac,
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
       {"cell": "A1", "branch": <edge_u>, "pos": <arc_frac>}

"""

import csv
import json
import logging
from dataclasses import dataclass, field, asdict
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

    # Postsynaptic side (POST_SYN cell) — empty string / -1 for contacts
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
      - presynaptic  location → cable mapping of the parent BOUTON  (A1B2)
      - postsynaptic location → cable mapping of the parent POST_SYN (A1B2P1)

    Contact resolution
    ------------------
    Each CONTACT segment (A1B2X1) defines a potential (unconfirmed) contact:
      - presynaptic  location → cable mapping of the parent BOUTON  (A1B2)
      - postsynaptic location → not available (no POST_SYN defined)
        → post_* fields are left as empty / -1

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
    ) -> Tuple[ConnectivityOutput, str, str]:
        """
        Build connectivity table and Arbor recipe.

        Args:
            registry      : SegmentRegistry from Phase 0.
            trees         : Phase 1 output — cell_name -> (tree, swc_path).
            mapping_table : Phase 3 output — CableMappingTable.
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
            connectivity, swc_paths, registry, recipe_path, syn_mechanism
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
                f"BOUTON {conn_row.bouton_name} — skipped"
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
            post_cell      = conn_row.post_cell_name
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
                f"BOUTON {contact_row.bouton_name} — skipped"
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
    ) -> None:
        """
        Write arbor_recipe.json.

        Schema
        ------
        {
          "cell_labels": {
            "A1": "swc/cell_A1.swc",
            ...
          },
          "synapses": [
            {
              "pre":  {"cell": "A1", "branch": <edge_u>, "pos": <arc_frac>},
              "post": {"cell": "A1B1P1", "branch": <edge_u>, "pos": <arc_frac>},
              "mechanism": "expsyn",
              "synapse_name": "A1B1P1S1",
              "bouton_name":  "A1B1"
            },
            ...
          ],
          "contacts": [
            {
              "pre":  {"cell": "A1", "branch": <edge_u>, "pos": <arc_frac>},
              "contact_name": "A1B1X1",
              "bouton_name":  "A1B1"
            },
            ...
          ]
        }
        """
        # Cell labels: all cells with a known SWC path
        cell_labels = {
            name: str(Path(swc_path).name)
            for name, swc_path in swc_paths.items()
        }

        synapses = []
        contacts = []

        for row in connectivity.rows:
            pre_loc = {
                "cell":   row.pre_cell,
                "branch": row.pre_edge_u,
                "pos":    round(row.pre_arc_frac, 6),
            }

            if row.connection_type == "synapse":
                post_loc = {
                    "cell":   row.post_cell,
                    "branch": row.post_edge_u,
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
                    "contact_name": row.synapse_name,   # contact_name stored here
                    "bouton_name":  row.bouton_name,
                })

        recipe = {
            "cell_labels": cell_labels,
            "synapses":    synapses,
            "contacts":    contacts,
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(recipe, f, indent=2)

        self.logger.info(
            f"Arbor recipe: {len(cell_labels)} cells, "
            f"{len(synapses)} synapses, {len(contacts)} contacts"
        )