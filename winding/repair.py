"""Mesh winding repair helpers for triangle meshes.

Shared contract for every public function below:

* Coordinates use a right-handed system.  A triangle is counter-clockwise when
  viewed from its front side, so its normal is
  ``normalize((p1 - p0) x (p2 - p0))``.
* A closed, consistently oriented mesh has outward-facing normals and a
  positive signed volume.
* Floating-point comparisons use an absolute tolerance of ``1e-6``.

A mesh is a mapping with exactly these keys::

    {
        "positions": [[x, y, z], ...],
        "faces":     [[i, j, k], ...],
    }

Each face is a triangle of three indices into ``positions``.  Violations of
this contract raise :class:`WindingError`.
"""

import math
from collections import Counter, deque

TOL = 1e-6


class WindingError(Exception):
    """Raised when a mesh or winding operation violates the documented contract."""


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _check_face(face) -> tuple:
    """Validate a face's structure (three integer-like indices)."""
    if not isinstance(face, (list, tuple)) or len(face) != 3:
        raise WindingError("each face must contain exactly three indices")
    if not all(_is_number(index) for index in face):
        raise WindingError("face indices must be integers")
    return tuple(face)


def _edge_key(a, b) -> frozenset:
    """Undirected edge identity, independent of endpoint order."""
    return frozenset((a, b))


def face_normal(p0, p1, p2) -> list[float]:
    """Return the unit face normal for the triangle ``(p0, p1, p2)``.

    The winding is counter-clockwise when viewed from the front, so the normal
    is ``normalize((p1 - p0) x (p2 - p0))``.  A degenerate triangle (zero area,
    collinear points, or coincident points) returns ``[0.0, 0.0, 0.0]`` as an
    explicit marker; NaN values are never returned.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    ux = p1[0] - p0[0]
    uy = p1[1] - p0[1]
    uz = p1[2] - p0[2]
    vx = p2[0] - p0[0]
    vy = p2[1] - p0[1]
    vz = p2[2] - p0[2]

    nx = uy * vz - uz * vy
    ny = uz * vx - ux * vz
    nz = ux * vy - uy * vx

    length = math.sqrt(nx * nx + ny * ny + nz * nz)
    if not math.isfinite(length) or length <= TOL:
        return [0.0, 0.0, 0.0]
    return [nx / length, ny / length, nz / length]


def signed_volume(faces, positions) -> float:
    """Return the signed volume enclosed by ``faces`` around the origin.

    The value is ``sum(dot(p0, cross(p1, p2))) / 6`` over all triangles.  It is
    positive when a closed mesh is consistently wound with outward-facing
    normals, and negative when the whole surface is inward-facing.

    Faces with repeated or out-of-range indices are invalid.  An empty face
    list or an empty position list raises :class:`WindingError` before any
    division is attempted.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    if not faces or not positions:
        raise WindingError("cannot measure the volume of an empty mesh")

    vertex_count = len(positions)
    total = 0.0
    for face in faces:
        i, j, k = _check_face(face)
        if not (0 <= i < vertex_count and 0 <= j < vertex_count and 0 <= k < vertex_count):
            raise WindingError("face index out of range")
        if len({i, j, k}) != 3:
            raise WindingError("face with repeated indices has no area")
        p0 = positions[i]
        p1 = positions[j]
        p2 = positions[k]
        cross_x = p1[1] * p2[2] - p1[2] * p2[1]
        cross_y = p1[2] * p2[0] - p1[0] * p2[2]
        cross_z = p1[0] * p2[1] - p1[1] * p2[0]
        total += p0[0] * cross_x + p0[1] * cross_y + p0[2] * cross_z
    return total / 6.0


def flip_face(face) -> tuple:
    """Reverse a triangle's winding by swapping its second and third vertices.

    The first vertex stays fixed.  The returned face is a tuple.  A face that
    does not contain exactly three indices raises :class:`WindingError`.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    a, b, c = _check_face(face)
    return (a, c, b)


def is_manifold(edges) -> bool:
    """Return whether every undirected edge has exactly two incident faces.

    ``edges`` is an iterable of ``(a, b)`` integer vertex pairs, one entry per
    face-edge incidence; ``(a, b)`` and ``(b, a)`` denote the same edge.  The
    result is ``True`` only when every edge occurs exactly twice.  An empty
    iterable is vacuously manifold and returns ``True``.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    counts = Counter()
    for edge in edges:
        a, b = edge
        counts[_edge_key(a, b)] += 1
    return all(count == 2 for count in counts.values())


def orient_connected(faces, positions) -> list[tuple]:
    """Make winding consistent inside every edge-connected component.

    The first face of each component keeps its input winding and seeds a
    breadth-first traversal.  Whenever a neighboring face traverses their
    shared edge in the same direction, that neighbor is flipped; the change is
    then propagated to the rest of the component.  Only edges shared by
    exactly two faces create adjacency.  Degenerate faces are returned
    unchanged and do not participate in adjacency.

    The returned list contains one tuple per input face in the original order.
    Invalid indices or malformed faces raise :class:`WindingError`; an empty
    face list returns ``[]``.  This function does not force a component to
    face outward; :func:`repair_mesh` performs that final global check.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    if not faces:
        return []

    vertex_count = len(positions)
    validated = []
    for face in faces:
        face = _check_face(face)
        if not all(0 <= index < vertex_count for index in face):
            raise WindingError("face index out of range")
        validated.append(face)

    # Degenerate faces (repeated indices or zero geometric area) are left
    # untouched and excluded from every adjacency structure.
    degenerate = [False] * len(validated)
    for face_index, face in enumerate(validated):
        if len(set(face)) != 3:
            degenerate[face_index] = True
            continue
        p0 = positions[face[0]]
        p1 = positions[face[1]]
        p2 = positions[face[2]]
        if face_normal(p0, p1, p2) == [0.0, 0.0, 0.0]:
            degenerate[face_index] = True

    # One entry per directed face-edge incidence: edge -> list of
    # (face_index, directed-edge endpoints).
    directed_incidences = {}
    undirected_counts = Counter()
    for face_index, face in enumerate(validated):
        if degenerate[face_index]:
            continue
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(start, end)
            undirected_counts[key] += 1
            directed_incidences.setdefault(key, []).append((face_index, start, end))

    # Adjacency exists only across edges shared by exactly two faces.
    adjacency = {face_index: [] for face_index in range(len(validated)) if not degenerate[face_index]}
    for key, incidences in directed_incidences.items():
        if undirected_counts[key] != 2 or len(incidences) != 2:
            continue
        first, second = incidences
        face_a, start_a, end_a = first
        face_b, start_b, end_b = second
        adjacency[face_a].append((face_b, start_a, end_a, start_b, end_b))
        adjacency[face_b].append((face_a, start_b, end_b, start_a, end_a))

    needs_flip = [False] * len(validated)
    visited = [False] * len(validated)

    # Traverse every edge-connected component, not just the first one.
    for seed in range(len(validated)):
        if degenerate[seed] or visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor, cur_start, cur_end, nb_start, nb_end in adjacency[current]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                # Consistent winding traverses a shared edge in opposite
                # directions.  Same-direction traversal means the neighbor
                # needs the opposite flip state from the current face; the
                # flip propagates onward through the BFS.
                if (cur_start, cur_end) == (nb_start, nb_end):
                    needs_flip[neighbor] = not needs_flip[current]
                else:
                    needs_flip[neighbor] = needs_flip[current]
                queue.append(neighbor)

    result = []
    for face_index, face in enumerate(validated):
        if degenerate[face_index] or not needs_flip[face_index]:
            result.append(face)
        else:
            result.append(flip_face(face))
    return result


def _check_mesh(mesh) -> tuple:
    """Validate the mesh container structure and return (positions, faces)."""
    if not isinstance(mesh, dict):
        raise WindingError("mesh must be a mapping")
    if set(mesh.keys()) != {"positions", "faces"}:
        raise WindingError("mesh must have exactly 'positions' and 'faces' keys")
    positions = mesh["positions"]
    faces = mesh["faces"]
    if not isinstance(positions, list) or not isinstance(faces, list):
        raise WindingError("positions and faces must be lists")
    for point in positions:
        if not isinstance(point, (list, tuple)) or len(point) != 3:
            raise WindingError("each position must be an [x, y, z] triple")
        if not all(_is_number(component) for component in point):
            raise WindingError("position components must be numbers")
    return positions, faces


def repair_mesh(mesh) -> dict:
    """Repair one mesh and report what changed.

    Degenerate faces are removed first.  Each remaining connected component is
    oriented consistently with :func:`orient_connected`; every closed
    component with negative signed volume is then flipped as a whole so its
    normals face outward.  A mesh is closed only when every undirected edge has
    exactly two incident faces (a bounding box is never used).

    The input is not mutated.  The return value is::

        {"faces": [(i, j, k), ...], "flipped": int, "closed": bool}

    ``flipped`` counts faces whose winding was reversed, not removed
    degenerate faces.  An empty mesh, a mesh left with no non-degenerate
    faces, an out-of-range index, or malformed input raises
    :class:`WindingError`.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    positions, raw_faces = _check_mesh(mesh)

    if not positions or not raw_faces:
        raise WindingError("cannot repair an empty mesh")

    vertex_count = len(positions)
    original_faces = []
    for face in raw_faces:
        face = _check_face(face)
        if not all(0 <= index < vertex_count for index in face):
            raise WindingError("face index out of range")
        original_faces.append(face)

    # Remove degenerate faces first so they can never zero out the volume.
    valid_faces = []
    for face in original_faces:
        if len(set(face)) != 3:
            continue
        p0 = positions[face[0]]
        p1 = positions[face[1]]
        p2 = positions[face[2]]
        if face_normal(p0, p1, p2) == [0.0, 0.0, 0.0]:
            continue
        valid_faces.append(face)

    if not valid_faces:
        raise WindingError("mesh has no non-degenerate faces")

    oriented = orient_connected(valid_faces, positions)

    # Closedness comes from edge incidence counts alone, never a bounding box.
    edge_counts = Counter()
    for face in oriented:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[_edge_key(start, end)] += 1
    closed = all(count == 2 for count in edge_counts.values())

    # Build edge-connected components among the surviving faces.
    component_incidences = {}
    component_edge_counts = Counter()
    for face_index, face in enumerate(oriented):
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            key = _edge_key(start, end)
            component_edge_counts[key] += 1
            component_incidences.setdefault(key, []).append(face_index)

    neighbors = {face_index: set() for face_index in range(len(oriented))}
    for key, incidences in component_incidences.items():
        if component_edge_counts[key] != 2 or len(incidences) != 2:
            continue
        a, b = incidences
        neighbors[a].add(b)
        neighbors[b].add(a)

    visited = [False] * len(oriented)
    components = []
    for seed in range(len(oriented)):
        if visited[seed]:
            continue
        visited[seed] = True
        members = []
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            members.append(current)
            for neighbor in neighbors[current]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        components.append(members)

    final_faces = list(oriented)
    for members in components:
        # A component is closed only when every one of its own edges has
        # exactly two incident faces.
        component_edges = {
            key
            for face_index in members
            for key in (
                _edge_key(*oriented[face_index][0:2]),
                _edge_key(*oriented[face_index][1:3]),
                _edge_key(oriented[face_index][2], oriented[face_index][0]),
            )
        }
        if any(component_edge_counts[key] != 2 for key in component_edges):
            continue
        component_faces = [oriented[face_index] for face_index in members]
        if signed_volume(component_faces, positions) < -TOL:
            for face_index in members:
                final_faces[face_index] = flip_face(oriented[face_index])

    flipped = sum(
        1 for original, final in zip(valid_faces, final_faces) if original != final
    )

    return {"faces": final_faces, "flipped": flipped, "closed": closed}


def validate_mesh(mesh) -> list[str]:
    """Return a list of contract violations found in ``mesh``.

    Possible entries are ``"non-manifold"`` (some edge does not have exactly
    two incident faces), ``"inward"`` (a closed component has negative signed
    volume), ``"degenerate"`` (a zero-area or repeated-index face), and
    ``"isolated-vertex"`` (a position unused by every face).  An empty mesh
    returns ``["empty"]``; an otherwise malformed mesh returns
    ``["invalid"]``.  A valid outward-facing mesh returns ``[]``.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    try:
        positions, raw_faces = _check_mesh(mesh)
    except WindingError:
        return ["invalid"]

    if not positions and not raw_faces:
        return ["empty"]

    vertex_count = len(positions)
    faces = []
    for face in raw_faces:
        try:
            face = _check_face(face)
        except WindingError:
            return ["invalid"]
        if not all(0 <= index < vertex_count for index in face):
            return ["invalid"]
        faces.append(face)

    violations = []

    # Degenerate faces: repeated indices or zero geometric area.
    degenerate_flags = []
    for face in faces:
        if len(set(face)) != 3:
            degenerate_flags.append(True)
            continue
        p0 = positions[face[0]]
        p1 = positions[face[1]]
        p2 = positions[face[2]]
        degenerate_flags.append(face_normal(p0, p1, p2) == [0.0, 0.0, 0.0])
    if any(degenerate_flags):
        violations.append("degenerate")

    # Manifoldness: every undirected edge has exactly two incident faces.
    edge_counts = Counter()
    for face in faces:
        for start, end in ((face[0], face[1]), (face[1], face[2]), (face[2], face[0])):
            edge_counts[_edge_key(start, end)] += 1
    if edge_counts and any(count != 2 for count in edge_counts.values()):
        violations.append("non-manifold")

    # Orientation: a closed component with negative signed volume faces inward.
    if faces and all(count == 2 for count in edge_counts.values()):
        try:
            if signed_volume(faces, positions) < -TOL:
                violations.append("inward")
        except WindingError:
            if "non-manifold" not in violations:
                violations.append("non-manifold")

    # Isolated vertices: positions never referenced by any face.
    used = {index for face in faces for index in face}
    if any(vertex not in used for vertex in range(vertex_count)):
        violations.append("isolated-vertex")

    return violations


def repair_batch(meshes, config) -> dict:
    """Repair several meshes in input order as one atomic batch.

    ``config`` is an optional settings mapping.  ``require_closed`` defaults to
    ``False``; when true, any open mesh makes the whole batch fail.  Every mesh
    is validated before any result is returned.  If one mesh is invalid, the
    batch raises :class:`WindingError` and exposes no partial result; input
    meshes are never mutated, so the caller can retry the batch unchanged.

    The return value is::

        {
            "meshes": [{"faces": [...], "flipped": int, "closed": bool}, ...],
            "count": len(meshes),
        }

    The order of ``"meshes"`` is exactly the input order, and each entry keeps
    its own flip count.

    Conventions: right-handed coordinates, outward-facing normals, tolerance
    1e-6.
    """
    if not isinstance(meshes, list):
        raise WindingError("batch input must be a list of meshes")
    config = config or {}
    require_closed = bool(config.get("require_closed", False))

    # Validate every mesh up front so a failure exposes no partial result.
    for mesh in meshes:
        violations = validate_mesh(mesh)
        if violations:
            raise WindingError(f"mesh failed validation: {violations}")

    # Each mesh gets its own repair call, so flip counts and orientation
    # state never leak from one mesh to the next.
    repaired = []
    for mesh in meshes:
        result = repair_mesh(mesh)
        if require_closed and not result["closed"]:
            raise WindingError("batch contains an open mesh")
        repaired.append(result)

    return {"meshes": repaired, "count": len(meshes)}
