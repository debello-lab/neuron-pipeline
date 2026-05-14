"""
Arbor recipe loader for reconstructed neuron network.

Usage:
    import arbor
    from arbor_recipe import ReconstructedRecipe
    recipe = ReconstructedRecipe("arbor_recipe.json")
    sim = arbor.simulation(recipe)

Note: synapse placements use '"root"' as a placeholder location.
For anatomically accurate placement, map pre_swc_u/post_swc_u from
connectivity.csv to Arbor branch indices -- load_swc_arbor branch
numbering differs from SWC row IDs.
"""
import json
import arbor


class ReconstructedRecipe(arbor.recipe):
    def __init__(self, recipe_json_path):
        super().__init__()
        with open(recipe_json_path) as f:
            self._data = json.load(f)
        self._cell_list = sorted(self._data["cell_labels"].keys())
        self._synapses = self._data["synapses"]
        # Index by post cell for O(1) lookup in cell_description / connections_on
        self._post_synapses: dict = {}
        for syn in self._synapses:
            self._post_synapses.setdefault(syn["post"]["cell"], []).append(syn)

    def num_cells(self):
        return len(self._cell_list)

    def cell_kind(self, gid):
        return arbor.cell_kind.cable

    def cell_description(self, gid):
        name = self._cell_list[gid]
        tree = arbor.load_swc_arbor(self._data["cell_labels"][name])
        dec = arbor.decor()

        for mech in self._data.get("cell_mechanisms", {}).get(name, []):
            dec.paint("(all)", arbor.density(mech))

        # Threshold detector so this cell can act as a pre-synaptic source.
        dec.place('"root"', arbor.threshold_detector(-10), "detector")

        # Place a synapse target for every incoming connection onto this cell.
        # Location '"root"' is a placeholder -- see module docstring.
        for syn in self._post_synapses.get(name, []):
            label = f'syn_{syn["synapse_name"]}'
            dec.place('"root"', arbor.synapse(syn["mechanism"]), label)

        return arbor.cable_cell(tree, dec)

    def connections_on(self, gid):
        name = self._cell_list[gid]
        conns = []
        for syn in self._post_synapses.get(name, []):
            src_gid = self._cell_list.index(syn["pre"]["cell"])
            target_label = f'syn_{syn["synapse_name"]}'
            conns.append(arbor.connection(
                (src_gid, "detector"),
                target_label,
                weight=1.0,
                delay=0.5,
            ))
        return conns
