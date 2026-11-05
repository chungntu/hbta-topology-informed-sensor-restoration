"""Drawing-checked 3D topology for the HBTA response sensors.

The HDF5 coordinates are interpreted together with ``Sensor layout.pdf``.
AG sensors lie on truss joints.  AL sensors lie at the midpoints of stringer
panels, rather than at cross-girder joints.  The graph is therefore a
sensor-projected joint/member graph: an edge can be a direct truss member or a
short structural path through an uninstrumented joint.  Both straight-line
distance and structural-path length are retained so the two are not confused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


RELATION_NAMES = (
    "truss_diagonal",
    "truss_chord",
    "deck_longitudinal",
    "deck_transverse",
    "deck_truss_interface",
    "lateral_bracing",
)


@dataclass(frozen=True)
class SpatialGraph:
    edges: np.ndarray
    edge_attributes: np.ndarray
    relation_ids: np.ndarray
    relation_names: tuple[str, ...]
    weights: np.ndarray
    structural_path_lengths: np.ndarray
    path_multiplicity: np.ndarray

    def subset(self, relation_names: set[str]) -> "SpatialGraph":
        relation_indices = {
            index
            for index, name in enumerate(self.relation_names)
            if name in relation_names
        }
        mask = np.isin(self.relation_ids, list(relation_indices))
        if not np.any(mask):
            raise ValueError(f"No edges found for relations {sorted(relation_names)}")
        return _make_graph(
            self.edges[mask],
            self.relation_ids[mask],
            self.edge_attributes[mask, :7],
            self.relation_names,
            self.structural_path_lengths[mask],
            self.path_multiplicity[mask],
        )


def _validate_inputs(
    sensor_names: np.ndarray, positions: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    names = np.asarray(sensor_names).astype(str)
    coordinates = np.asarray(positions, dtype=np.float64)
    if names.ndim != 1:
        raise ValueError("sensor_names must be one-dimensional")
    if coordinates.shape != (names.size, 3):
        raise ValueError(
            f"positions must have shape ({names.size}, 3), got {coordinates.shape}"
        )
    if len(set(names)) != names.size:
        raise ValueError("sensor_names must be unique")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("positions contain non-finite coordinates")
    return names, coordinates


def _relation_balanced_inverse_length_weights(
    lengths: np.ndarray, relation_ids: np.ndarray
) -> np.ndarray:
    """Give each relation equal total influence and favor shorter load paths."""
    weights = np.empty(lengths.size, dtype=np.float64)
    relations = np.unique(relation_ids)
    for relation in relations:
        mask = relation_ids == relation
        inverse_length = 1.0 / np.maximum(lengths[mask], 1e-12)
        inverse_length /= inverse_length.mean()
        weights[mask] = inverse_length / mask.sum()
    weights *= weights.size / len(relations)
    return weights


def _make_graph(
    edges: np.ndarray,
    relation_ids: np.ndarray,
    geometric_attributes: np.ndarray,
    relation_names: tuple[str, ...] = RELATION_NAMES,
    structural_path_lengths: np.ndarray | None = None,
    path_multiplicity: np.ndarray | None = None,
) -> SpatialGraph:
    edges = np.asarray(edges, dtype=np.int64)
    relation_ids = np.asarray(relation_ids, dtype=np.int64)
    geometric_attributes = np.asarray(geometric_attributes, dtype=np.float64)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (E, 2)")
    if relation_ids.shape != (edges.shape[0],):
        raise ValueError("relation_ids must have shape (E,)")
    if geometric_attributes.shape != (edges.shape[0], 7):
        raise ValueError("geometric_attributes must have shape (E, 7)")
    if structural_path_lengths is None:
        structural_path_lengths = geometric_attributes[:, 3].copy()
    structural_path_lengths = np.asarray(
        structural_path_lengths, dtype=np.float64
    )
    if structural_path_lengths.shape != (edges.shape[0],):
        raise ValueError("structural_path_lengths must have shape (E,)")
    if np.any(structural_path_lengths <= 0.0):
        raise ValueError("structural_path_lengths must be positive")
    if path_multiplicity is None:
        path_multiplicity = np.ones(edges.shape[0], dtype=np.float64)
    path_multiplicity = np.asarray(path_multiplicity, dtype=np.float64)
    if path_multiplicity.shape != (edges.shape[0],):
        raise ValueError("path_multiplicity must have shape (E,)")
    one_hot = np.eye(len(relation_names), dtype=np.float64)[relation_ids]
    attributes = np.concatenate(
        [
            geometric_attributes,
            structural_path_lengths[:, None],
            path_multiplicity[:, None],
            one_hot,
        ],
        axis=1,
    )
    weights = _relation_balanced_inverse_length_weights(
        structural_path_lengths, relation_ids
    )
    return SpatialGraph(
        edges=edges,
        edge_attributes=attributes,
        relation_ids=relation_ids,
        relation_names=relation_names,
        weights=weights,
        structural_path_lengths=structural_path_lengths,
        path_multiplicity=path_multiplicity,
    )


def _geometric_attributes(positions: np.ndarray, edges: np.ndarray) -> np.ndarray:
    delta = positions[edges[:, 1]] - positions[edges[:, 0]]
    length = np.linalg.norm(delta, axis=1)
    if np.any(length <= 0.0):
        raise ValueError("Topology contains a zero-length edge")
    direction = delta / length[:, None]
    return np.column_stack([delta, length, direction])


def _drawing_structural_paths(
    positions: np.ndarray,
    edges: np.ndarray,
    relation_ids: np.ndarray,
    relation_names: tuple[str, ...] = RELATION_NAMES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return member-following path length and number of equivalent paths.

    The AL sensors are at stringer-panel midpoints.  Two AL sensors at the
    same longitudinal coordinate are therefore coupled through either end
    cross girder, not by a fictitious transverse member at the sensor sites.
    """
    direct = np.linalg.norm(
        positions[edges[:, 1]] - positions[edges[:, 0]], axis=1
    )
    paths = direct.copy()
    multiplicity = np.ones(edges.shape[0], dtype=np.float64)
    al = np.flatnonzero(
        np.isclose(positions[:, 2], 0.19, atol=0.05)
    )
    al_x = np.unique(np.round(positions[al, 0], decimals=6))
    panel_length = float(np.median(np.diff(al_x)))

    for edge_index, relation_id in enumerate(relation_ids):
        relation = relation_names[int(relation_id)]
        left, right = edges[edge_index]
        delta = np.abs(positions[right] - positions[left])
        if relation == "deck_transverse":
            # Half a panel to a cross girder, transverse to the adjacent
            # stringer, and half a panel back.  The two panel ends give two
            # equivalent load paths.
            paths[edge_index] = panel_length + delta[1]
            multiplicity[edge_index] = 2.0
        elif relation == "deck_truss_interface":
            # From an AL panel midpoint to its end joint along the stringer,
            # then through the cross girder to the deck-level wall joint.
            paths[edge_index] = delta[0] + np.hypot(delta[1], delta[2])
        elif relation == "truss_chord" and delta[0] > panel_length + 1e-6:
            # The two instrumented chord joints have one uninstrumented joint
            # between them.  Use the complementary wall sensor at the middle
            # x-coordinate to recover the arch height when available.
            midpoint_x = 0.5 * (
                positions[left, 0] + positions[right, 0]
            )
            candidates = np.flatnonzero(
                np.isclose(positions[:, 0], midpoint_x)
                & (positions[:, 2] > 1.0)
            )
            if candidates.size and positions[left, 2] > 1.0:
                midpoint = np.asarray(
                    [midpoint_x, positions[left, 1], positions[candidates[0], 2]]
                )
                paths[edge_index] = (
                    np.linalg.norm(midpoint - positions[left])
                    + np.linalg.norm(positions[right] - midpoint)
                )
    return paths, multiplicity


def build_hbta_spatial_graph(
    sensor_names: np.ndarray, positions: np.ndarray
) -> SpatialGraph:
    """Build a drawing-checked sensor projection of the joint/member graph.

    AG wall edges follow diagonal members or chord paths.  AL longitudinal
    edges join neighboring stringer-panel midpoint sensors through their
    common end joint.  AL transverse edges represent the two cross-girder
    paths between adjacent parallel stringers.  A deck-level AG joint connects
    to the two AL panels incident on that joint.
    """
    names, coordinates = _validate_inputs(sensor_names, positions)
    index = {name: position for position, name in enumerate(names)}
    ag_names = [name for name in names if name.startswith("AG")]
    al_names = [name for name in names if name.startswith("AL")]
    if len(ag_names) != 18 or len(al_names) != 40:
        raise ValueError(
            "HBTA spatial topology expects 18 AG and 40 AL response sensors"
        )

    edge_relations: dict[tuple[int, int], int] = {}

    def add(left: int, right: int, relation: str) -> None:
        edge = tuple(sorted((int(left), int(right))))
        relation_id = RELATION_NAMES.index(relation)
        previous = edge_relations.get(edge)
        if previous is not None and previous != relation_id:
            raise ValueError(f"Edge {edge} assigned to multiple relations")
        edge_relations[edge] = relation_id

    ag_indices = np.asarray([index[name] for name in ag_names])
    for side_sign in (1.0, -1.0):
        wall = ag_indices[np.sign(coordinates[ag_indices, 1]) == side_sign]
        wall = wall[np.argsort(coordinates[wall, 0])]
        if wall.size != 9:
            raise ValueError("Expected nine AG sensors on each truss wall")
        for left, right in zip(wall[:-1], wall[1:]):
            add(left, right, "truss_diagonal")
        for left, right in zip(wall[:-2], wall[2:]):
            add(left, right, "truss_chord")

    al_indices = np.asarray([index[name] for name in al_names])
    unique_x = np.unique(np.round(coordinates[al_indices, 0], decimals=6))
    unique_y = np.unique(np.round(coordinates[al_indices, 1], decimals=6))
    if unique_x.size != 10 or unique_y.size != 4:
        raise ValueError("Expected the AL sensors to form a 10 x 4 deck grid")

    for y in unique_y:
        row = al_indices[np.isclose(coordinates[al_indices, 1], y)]
        row = row[np.argsort(coordinates[row, 0])]
        for left, right in zip(row[:-1], row[1:]):
            add(left, right, "deck_longitudinal")
    for x in unique_x:
        column = al_indices[np.isclose(coordinates[al_indices, 0], x)]
        column = column[np.argsort(coordinates[column, 1])]
        for left, right in zip(column[:-1], column[1:]):
            add(left, right, "deck_transverse")

    deck_level = np.min(coordinates[ag_indices, 2])
    low_ag = ag_indices[np.isclose(coordinates[ag_indices, 2], deck_level, atol=0.5)]
    if low_ag.size != 9:
        raise ValueError("Expected nine deck-level AG sensors")
    for ag in low_ag:
        side_sign = np.sign(coordinates[ag, 1])
        outer_y = (
            np.max(coordinates[al_indices, 1])
            if side_sign > 0
            else np.min(coordinates[al_indices, 1])
        )
        outer_row = al_indices[np.isclose(coordinates[al_indices, 1], outer_y)]
        x_distance = np.abs(coordinates[outer_row, 0] - coordinates[ag, 0])
        for al in outer_row[np.argsort(x_distance)[:2]]:
            add(ag, al, "deck_truss_interface")

    # The plan view in Drawings.pdf shows X-bracing in every deck panel.
    # Deck-level AG sensors alternate between the north and south walls at
    # consecutive interior panel joints.  Consecutive sensors in x therefore
    # coincide with one directly observed diagonal of each interior X-panel.
    low_ag_by_x = low_ag[np.argsort(coordinates[low_ag, 0])]
    for left, right in zip(low_ag_by_x[:-1], low_ag_by_x[1:]):
        if np.sign(coordinates[left, 1]) == np.sign(coordinates[right, 1]):
            raise ValueError("Lateral-bracing endpoints must be on opposite walls")
        add(left, right, "lateral_bracing")

    edges = np.asarray(sorted(edge_relations), dtype=np.int64)
    relation_ids = np.asarray([edge_relations[tuple(edge)] for edge in edges])
    structural_paths, multiplicity = _drawing_structural_paths(
        coordinates, edges, relation_ids
    )
    return _make_graph(
        edges,
        relation_ids,
        _geometric_attributes(coordinates, edges),
        structural_path_lengths=structural_paths,
        path_multiplicity=multiplicity,
    )


def graph_from_edges(
    sensor_names: np.ndarray,
    positions: np.ndarray,
    edges: np.ndarray,
    relation_ids: np.ndarray,
    relation_names: tuple[str, ...] = RELATION_NAMES,
) -> SpatialGraph:
    """Recompute 3D attributes and weights for a control edge set."""
    _, coordinates = _validate_inputs(sensor_names, positions)
    edges = np.sort(np.asarray(edges, dtype=np.int64), axis=1)
    structural_paths, multiplicity = _drawing_structural_paths(
        coordinates, edges, relation_ids, relation_names
    )
    return _make_graph(
        edges,
        relation_ids,
        _geometric_attributes(coordinates, edges),
        relation_names,
        structural_paths,
        multiplicity,
    )


def permuted_spatial_control(
    graph: SpatialGraph,
    sensor_names: np.ndarray,
    positions: np.ndarray,
    seed: int,
) -> SpatialGraph:
    """Permute sensor labels within AG and AL families, preserving relations."""
    names, _ = _validate_inputs(sensor_names, positions)
    rng = np.random.default_rng(seed)
    mapping = np.arange(names.size)
    for prefix in ("AG", "AL"):
        nodes = np.flatnonzero(np.char.startswith(names, prefix))
        mapping[nodes] = rng.permutation(nodes)
    edges = np.sort(mapping[graph.edges], axis=1)
    return graph_from_edges(
        names, positions, edges, graph.relation_ids, graph.relation_names
    )


def random_spatial_control(
    graph: SpatialGraph,
    sensor_names: np.ndarray,
    positions: np.ndarray,
    seed: int,
) -> SpatialGraph:
    """Randomize edges within each relation's valid sensor/axis scope."""
    names, coordinates = _validate_inputs(sensor_names, positions)
    rng = np.random.default_rng(seed)
    ag = np.flatnonzero(np.char.startswith(names, "AG"))
    al = np.flatnonzero(np.char.startswith(names, "AL"))
    deck_level = np.min(coordinates[ag, 2])
    low_ag = ag[np.isclose(coordinates[ag, 2], deck_level, atol=0.5)]
    original = {tuple(edge) for edge in graph.edges.tolist()}
    used = set(original)
    selected_edges: list[tuple[int, int]] = []
    selected_relations: list[int] = []

    for relation_id, relation in enumerate(graph.relation_names):
        count = int(np.sum(graph.relation_ids == relation_id))
        if count == 0:
            continue
        if relation in {"truss_diagonal", "truss_chord"}:
            candidates = []
            for side_sign in (1.0, -1.0):
                wall = ag[np.sign(coordinates[ag, 1]) == side_sign]
                candidates.extend(
                    (int(left), int(right))
                    for position, left in enumerate(wall)
                    for right in wall[position + 1 :]
                )
        elif relation == "deck_longitudinal":
            candidates = [
                (int(left), int(right))
                for position, left in enumerate(al)
                for right in al[position + 1 :]
                if np.isclose(coordinates[left, 1], coordinates[right, 1])
            ]
        elif relation == "deck_transverse":
            candidates = [
                (int(left), int(right))
                for position, left in enumerate(al)
                for right in al[position + 1 :]
                if np.isclose(coordinates[left, 0], coordinates[right, 0])
            ]
        elif relation == "deck_truss_interface":
            candidates = []
            for left in low_ag:
                side_sign = np.sign(coordinates[left, 1])
                outer_y = (
                    np.max(coordinates[al, 1])
                    if side_sign > 0
                    else np.min(coordinates[al, 1])
                )
                outer_row = al[np.isclose(coordinates[al, 1], outer_y)]
                candidates.extend((int(left), int(right)) for right in outer_row)
        elif relation == "lateral_bracing":
            candidates = [
                (int(left), int(right))
                for position, left in enumerate(low_ag)
                for right in low_ag[position + 1 :]
                if np.sign(coordinates[left, 1])
                != np.sign(coordinates[right, 1])
            ]
        else:
            raise ValueError(f"No random-control scope for relation {relation}")

        candidates = sorted(
            {
                tuple(sorted(edge))
                for edge in candidates
                if tuple(sorted(edge)) not in used
            }
        )
        if len(candidates) < count:
            raise ValueError(
                f"Only {len(candidates)} controls available for {count} {relation} edges"
            )
        choice = rng.choice(len(candidates), size=count, replace=False)
        chosen = [candidates[position] for position in choice]
        selected_edges.extend(chosen)
        selected_relations.extend([relation_id] * count)
        used.update(chosen)

    order = np.lexsort(
        (
            np.asarray(selected_edges)[:, 1],
            np.asarray(selected_edges)[:, 0],
        )
    )
    edges = np.asarray(selected_edges, dtype=np.int64)[order]
    relation_ids = np.asarray(selected_relations, dtype=np.int64)[order]
    return graph_from_edges(
        names, coordinates, edges, relation_ids, graph.relation_names
    )


def graph_audit(
    graph: SpatialGraph, sensor_names: np.ndarray
) -> dict[str, object]:
    names = np.asarray(sensor_names).astype(str)
    relation_counts = {
        relation: int(np.sum(graph.relation_ids == relation_id))
        for relation_id, relation in enumerate(graph.relation_names)
    }
    family_counts: dict[str, int] = {}
    for left, right in graph.edges:
        family = "-".join(sorted((names[left][:2], names[right][:2])))
        family_counts[family] = family_counts.get(family, 0) + 1
    lengths = graph.edge_attributes[:, 3]
    structural_lengths = graph.structural_path_lengths
    return {
        "edge_count": int(graph.edges.shape[0]),
        "edge_attribute_columns": [
            "dx",
            "dy",
            "dz",
            "length",
            "unit_dx",
            "unit_dy",
            "unit_dz",
            "structural_path_length",
            "parallel_path_count",
            *[f"relation_{name}" for name in graph.relation_names],
        ],
        "relation_counts": relation_counts,
        "sensor_family_counts": family_counts,
        "length_min": float(lengths.min()),
        "length_median": float(np.median(lengths)),
        "length_max": float(lengths.max()),
        "structural_path_length_min": float(structural_lengths.min()),
        "structural_path_length_median": float(
            np.median(structural_lengths)
        ),
        "structural_path_length_max": float(structural_lengths.max()),
        "two_path_edge_count": int(np.sum(graph.path_multiplicity == 2.0)),
        "weight_mean": float(graph.weights.mean()),
        "drawing_source": "Sensor layout.pdf and Drawings.pdf in the external HBTA raw-data directory",
    }
