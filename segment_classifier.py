from VastControlClass_exporting import VASTControlClass
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, NamedTuple
import re

@dataclass
class SegmentInfo:
    seg_id: int
    name: str
    role: str  # AXON | BOUTON | POST_SYN | SYNAPSE | CONTACT | UNKNOWN
    bbox: Tuple[int, int, int, int, int, int]  # x1, y1, z1, x2, y2, z2


class ConnectivityRow(NamedTuple):
    axon_name: str
    axon_seg_id: int
    bouton_name: str
    bouton_seg_id: int
    post_syn_name: str
    post_syn_seg_id: int
    synapse_name: str
    synapse_seg_id: int


class ContactRow(NamedTuple):
    axon_name: str
    axon_seg_id: int
    bouton_name: str
    bouton_seg_id: int
    contact_name: str
    contact_seg_id: int


@dataclass
class SegmentRegistry:
    segments: Dict[str, SegmentInfo] = field(default_factory=dict)
    connectivity: List[ConnectivityRow] = field(default_factory=list)
    contacts: List[ContactRow] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class SegmentClassifier:
    def __init__(self):
        self.vast = VASTControlClass()
        self.vast.connect("localhost", 22081, 5)

        # Pre-compiled patterns (most-specific first for the if/elif chain)
        self.patterns = {
            'SYNAPSE':  re.compile(r'^(A\d+)(B\d+)(P\d+)(S\d+)$'),
            'CONTACT':  re.compile(r'^(A\d+)(B\d+)(X\d+)$'),
            'POST_SYN': re.compile(r'^(A\d+)(B\d+)(P\d+)$'),
            'BOUTON':   re.compile(r'^(A\d+)(B\d+)$'),
            'AXON':     re.compile(r'^(A\d+)$'),
        }

    def classify_segments(self) -> SegmentRegistry:
        """Classify all VAST segments and build connectivity / contact tables.

        Returns:
            SegmentRegistry with classified segments, wiring rows, contact rows,
            and validation warnings.
        """
        # 1. Fetch names and metadata in one pass each
        segment_names = self.vast.get_all_segment_names()  # List[str]
        segment_data = self.vast.get_all_segment_data()    # List[Dict]

        if not segment_names or not segment_data:
            raise RuntimeError("Failed to retrieve segment data from VAST")

        registry = SegmentRegistry()

        # --- Pass 1: classify every segment ----------------------
        for i, name in enumerate(segment_names):
            if i == 0 or not name:  # skip background / empty names
                continue

            role = 'UNKNOWN'
            for role_name, pattern in self.patterns.items():
                if pattern.match(name):
                    role = role_name
                    break

            if role == 'UNKNOWN':
                registry.warnings.append(f"Unknown segment name: {name} (id={i})")

            bbox = tuple(segment_data[i]['boundingbox']) if i < len(segment_data) else ()
            registry.segments[name] = SegmentInfo(seg_id=i, name=name, role=role, bbox=bbox)

        # Helper: look up seg_id by name (-1 if missing)
        def get_id(name: str) -> int:
            info = registry.segments.get(name)
            return info.seg_id if info else -1

        # --- Pass 2: build connectivity and contact tables ----------------------
        for name, info in registry.segments.items():

            if info.role == 'SYNAPSE':
                m = self.patterns['SYNAPSE'].match(name)
                g = m.groups()  # ('A1', 'B1', 'P1', 'S1')
                axon_name    = g[0]
                bouton_name  = g[0] + g[1]
                post_syn_name = g[0] + g[1] + g[2]

                registry.connectivity.append(ConnectivityRow(
                    axon_name=axon_name,       axon_seg_id=get_id(axon_name),
                    bouton_name=bouton_name,   bouton_seg_id=get_id(bouton_name),
                    post_syn_name=post_syn_name, post_syn_seg_id=get_id(post_syn_name),
                    synapse_name=name,         synapse_seg_id=info.seg_id,
                ))

            elif info.role == 'CONTACT':
                m = self.patterns['CONTACT'].match(name)
                g = m.groups()  # ('A6', 'B1', 'X1')
                axon_name   = g[0]
                bouton_name = g[0] + g[1]

                registry.contacts.append(ContactRow(
                    axon_name=axon_name,     axon_seg_id=get_id(axon_name),
                    bouton_name=bouton_name, bouton_seg_id=get_id(bouton_name),
                    contact_name=name,       contact_seg_id=info.seg_id,
                ))

        # --- Pass 3: cross-check names vs VAST hierarchy -------------------------
        # Each segment_data entry has hierarchy[0] = parent seg_id.
        # Derive the expected parent from the name and compare.
        vast_parent_of = {}  # seg_id -> parent seg_id in VAST tree
        for sd in segment_data:
            vast_parent_of[sd['id']] = sd['hierarchy'][0]

        # Map each non-AXON segment to its name-derived parent name
        name_derived_parent = {}
        for name, info in registry.segments.items():
            if info.role == 'BOUTON':
                m = self.patterns['BOUTON'].match(name)
                name_derived_parent[name] = m.group(1)           # AXON
            elif info.role == 'POST_SYN':
                m = self.patterns['POST_SYN'].match(name)
                name_derived_parent[name] = m.group(1) + m.group(2)  # BOUTON
            elif info.role == 'SYNAPSE':
                m = self.patterns['SYNAPSE'].match(name)
                name_derived_parent[name] = m.group(1) + m.group(2) + m.group(3)  # POST_SYN
            elif info.role == 'CONTACT':
                m = self.patterns['CONTACT'].match(name)
                name_derived_parent[name] = m.group(1) + m.group(2)  # BOUTON

        for name, expected_parent_name in name_derived_parent.items():
            info = registry.segments[name]
            expected_parent_id = get_id(expected_parent_name)
            vast_parent_id = vast_parent_of.get(info.seg_id, 0)

            if expected_parent_id != -1 and vast_parent_id != expected_parent_id:
                vast_parent_name = (
                    segment_names[vast_parent_id]
                    if 0 < vast_parent_id < len(segment_names)
                    else f"seg#{vast_parent_id}"
                )
                registry.warnings.append(
                    f"HIERARCHY MISMATCH: {name} -- name implies parent "
                    f"{expected_parent_name}, but VAST tree says parent is "
                    f"{vast_parent_name} (id={vast_parent_id})")

        # --- Pass 4: validation -------------------------------------------------
        # Boutons that are referenced by at least one child
        boutons_with_children = (
            {r.bouton_name for r in registry.connectivity}
            | {r.bouton_name for r in registry.contacts}
        )
        # Post-synaptic segments that are referenced by at least one synapse
        post_syns_with_synapse = {r.post_syn_name for r in registry.connectivity}

        for name, info in registry.segments.items():
            if info.role == 'BOUTON' and name not in boutons_with_children:
                registry.warnings.append(
                    f"BOUTON {name} has no children (no P#/S# or X# segments)")
            if info.role == 'POST_SYN' and name not in post_syns_with_synapse:
                registry.warnings.append(
                    f"POST_SYN {name} has no SYNAPSE child")

        for row in registry.connectivity:
            if row.axon_seg_id == -1:
                registry.warnings.append(
                    f"SYNAPSE {row.synapse_name} references non-existent AXON {row.axon_name}")
            if row.bouton_seg_id == -1:
                registry.warnings.append(
                    f"SYNAPSE {row.synapse_name} references non-existent BOUTON {row.bouton_name}")
            if row.post_syn_seg_id == -1:
                registry.warnings.append(
                    f"SYNAPSE {row.synapse_name} references non-existent POST_SYN {row.post_syn_name}")

        for row in registry.contacts:
            if row.axon_seg_id == -1:
                registry.warnings.append(
                    f"CONTACT {row.contact_name} references non-existent AXON {row.axon_name}")
            if row.bouton_seg_id == -1:
                registry.warnings.append(
                    f"CONTACT {row.contact_name} references non-existent BOUTON {row.bouton_name}")

        # --- Summary -------------------------------------------------
        role_counts = {}
        for info in registry.segments.values():
            role_counts[info.role] = role_counts.get(info.role, 0) + 1

        print(f"Classified {len(registry.segments)} segments:")
        for role in ('AXON', 'BOUTON', 'POST_SYN', 'SYNAPSE', 'CONTACT', 'UNKNOWN'):
            if role_counts.get(role, 0):
                print(f"  {role}: {role_counts[role]}")
        print(f"  Connectivity rows: {len(registry.connectivity)}")
        print(f"  Contact rows:      {len(registry.contacts)}")
        if registry.warnings:
            print(f"  Warnings:          {len(registry.warnings)}")
            for w in registry.warnings:
                print(f"    - {w}")

        return registry


if __name__ == "__main__":
    classifier = SegmentClassifier()
    registry = classifier.classify_segments()
