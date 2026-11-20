"""Regression tests for mesh winding repair.

The implementation in ``winding/repair.py`` is intentionally all stubs, so this
suite starts red: every test calls at least one stubbed function and fails with
``NotImplementedError`` until the ten defects are fixed.
"""

import math

import pytest

from winding import (
    WindingError,
    face_normal,
    flip_face,
    is_manifold,
    orient_connected,
    repair_batch,
    repair_mesh,
    signed_volume,
    validate_mesh,
)

TOL = 1e-6


def _mesh(positions, faces):
    return {"positions": positions, "faces": faces}


def _cube():
    """Unit cube with counter-clockwise, outward-facing triangles."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ]
    return _mesh(positions, faces)


def _tetrahedron():
    """Unit-corner tetrahedron with outward faces and volume 1/6."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    faces = [
        [0, 1, 3],
        [0, 3, 2],
        [1, 2, 3],
        [0, 2, 1],
    ]
    return _mesh(positions, faces)


def _tetrahedron_with_last_face_flipped():
    mesh = _tetrahedron()
    mesh["faces"][-1] = [0, 1, 2]
    return mesh


def _open_quad():
    """Two triangles sharing one edge, with a non-zero bounding box."""
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0],
    ]
    faces = [[0, 1, 2], [1, 3, 2]]
    return _mesh(positions, faces)


def _snapshot(mesh):
    return tuple(tuple(face) for face in mesh["faces"])


def test_face_normal_follows_right_hand_rule():
    normal = face_normal([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0])

    assert normal == pytest.approx([0.0, 0.0, 1.0], abs=TOL)
    assert math.sqrt(sum(component * component for component in normal)) == pytest.approx(
        1.0, abs=TOL
    )


def test_signed_volume_divides_by_six():
    mesh = _tetrahedron()

    volume = signed_volume(mesh["faces"], mesh["positions"])

    assert volume == pytest.approx(1.0 / 6.0, abs=TOL)


def test_flip_face_swaps_second_and_third_vertices():
    assert flip_face([3, 1, 2]) == (3, 2, 1)


def test_is_manifold_rejects_edge_with_three_faces():
    tetrahedron_edges = [
        (0, 1),
        (1, 3),
        (3, 0),
        (0, 3),
        (3, 2),
        (2, 0),
        (1, 2),
        (2, 3),
        (3, 1),
        (0, 2),
        (2, 1),
        (1, 0),
    ]
    duplicate_face_edges = [(0, 1), (1, 3), (3, 0)]
    manifold_strip = [(0, 1), (0, 1), (1, 2), (1, 2)]

    assert is_manifold(tetrahedron_edges + duplicate_face_edges) is False
    assert is_manifold(manifold_strip) is True


def test_orient_connected_propagates_flip_to_neighbor():
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ]
    faces = [(0, 1, 2), (1, 2, 3)]

    oriented = orient_connected(faces, positions)

    assert oriented == [(0, 1, 2), (1, 3, 2)]


def test_orient_connected_visits_every_component():
    positions = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ]
    faces = [
        (0, 1, 2),
        (1, 2, 3),
        (4, 5, 6),
        (5, 6, 7),
    ]

    oriented = orient_connected(faces, positions)

    assert oriented == [
        (0, 1, 2),
        (1, 3, 2),
        (4, 5, 6),
        (5, 7, 6),
    ]


def test_open_surface_is_not_reported_closed():
    result = repair_mesh(_open_quad())

    assert result["closed"] is False
    assert len(result["faces"]) == 2


def test_repair_mesh_removes_degenerate_faces():
    mesh = _tetrahedron()
    mesh["faces"].append([0, 0, 1])

    result = repair_mesh(mesh)

    assert len(result["faces"]) == 4
    assert all(len(set(face)) == 3 for face in result["faces"])
    assert result["closed"] is True
    assert signed_volume(result["faces"], mesh["positions"]) == pytest.approx(
        1.0 / 6.0, abs=TOL
    )


def test_repair_batch_keeps_flip_counts_separate():
    clean = _tetrahedron()
    repaired = _tetrahedron_with_last_face_flipped()

    result = repair_batch([repaired, clean], {})

    assert result["count"] == 2
    assert result["meshes"][0]["flipped"] == 1
    assert result["meshes"][1]["flipped"] == 0


def test_empty_faces_cannot_be_measured():
    with pytest.raises(WindingError):
        signed_volume([], [])


def test_unit_cube_repairs_to_volume_one():
    mesh = _cube()
    mesh["faces"] = [face[::-1] for face in mesh["faces"]]

    result = repair_mesh(mesh)

    assert result["closed"] is True
    assert signed_volume(result["faces"], mesh["positions"]) == pytest.approx(
        1.0, abs=TOL
    )


def test_correct_cube_needs_no_flips():
    mesh = _cube()

    assert validate_mesh(mesh) == []
    result = repair_mesh(mesh)

    assert result["closed"] is True
    assert result["flipped"] == 0
    assert result["faces"] == [tuple(face) for face in mesh["faces"]]


def test_empty_mesh_raises_winding_error():
    empty = {"positions": [], "faces": []}

    with pytest.raises(WindingError):
        repair_mesh(empty)

    good = _tetrahedron_with_last_face_flipped()
    before = _snapshot(good)
    with pytest.raises(WindingError):
        repair_batch([good, empty], {})
    assert _snapshot(good) == before


def test_single_triangle_is_not_closed():
    mesh = _mesh(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 1.0]],
        [[0, 1, 2]],
    )

    result = repair_mesh(mesh)

    assert result["closed"] is False
    assert len(result["faces"]) == 1


def test_batch_success_preserves_input_order():
    cube = _cube()
    tetra = _tetrahedron()

    result = repair_batch([cube, tetra], {})

    assert result["count"] == 2
    assert [len(item["faces"]) for item in result["meshes"]] == [12, 4]
    assert [item["closed"] for item in result["meshes"]] == [True, True]
