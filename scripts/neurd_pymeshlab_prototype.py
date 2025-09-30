"""Stage 1 prototype for replacing Meshlab CLI decimation with pymeshlab.

This script loads the NEURD integration fixture mesh, runs the equivalent
quadric edge-collapse decimator implemented in pymeshlab, and prints summary
metrics so we can compare drift against the legacy pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import pymeshlab
import trimesh


FIXTURE_SEGMENT_ID = "864691135510518224"
FIXTURE_MESH = Path("external/NEURD/tests/fixtures") / f"{FIXTURE_SEGMENT_ID}.off"


@dataclass
class MeshStats:
    vertex_count: int
    face_count: int
    surface_area: float
    bbox_diagonal: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "vertex_count": self.vertex_count,
            "face_count": self.face_count,
            "surface_area": self.surface_area,
            "bbox_diagonal": self.bbox_diagonal,
        }


def load_mesh(mesh_path: Path) -> trimesh.Trimesh:
    if not mesh_path.exists():
        raise FileNotFoundError(f"Fixture mesh not found: {mesh_path}")
    return trimesh.load(mesh_path, process=False)


def trimesh_stats(mesh: trimesh.Trimesh) -> MeshStats:
    bbox_extents = mesh.bounding_box.extents
    bbox_diagonal = float((bbox_extents ** 2).sum() ** 0.5)
    return MeshStats(
        vertex_count=int(mesh.vertices.shape[0]),
        face_count=int(mesh.faces.shape[0]),
        surface_area=float(mesh.area),
        bbox_diagonal=bbox_diagonal,
    )


def decimate_with_pymeshlab(
    mesh: trimesh.Trimesh, decimation_ratio: float = 0.25
) -> trimesh.Trimesh:
    ms = pymeshlab.MeshSet()
    ms.add_mesh(
        pymeshlab.Mesh(
            vertex_matrix=mesh.vertices.astype(float),
            face_matrix=mesh.faces.astype(int),
        ),
        mesh_name="original",
    )

    ms.meshing_remove_duplicate_vertices()
    ms.meshing_decimation_quadric_edge_collapse(
        targetperc=decimation_ratio,
        targetfacenum=100000,
        qualitythr=1.0,
        preserveboundary=True,
        boundaryweight=1.0,
        preservenormal=True,
        preservetopology=True,
        optimalplacement=True,
        planarquadric=True,
        planarweight=1.0,
        qualityweight=False,
        autoclean=True,
        selected=False,
    )

    decimated = ms.current_mesh()
    return trimesh.Trimesh(
        vertices=decimated.vertex_matrix(),
        faces=decimated.face_matrix(),
        process=False,
    )


def compare_stats(original: MeshStats, decimated: MeshStats) -> Dict[str, float]:
    return {
        "vertex_ratio": decimated.vertex_count / original.vertex_count,
        "face_ratio": decimated.face_count / original.face_count,
        "area_delta_pct": (decimated.surface_area - original.surface_area)
        / original.surface_area
        * 100,
        "bbox_diagonal_delta_pct": (decimated.bbox_diagonal - original.bbox_diagonal)
        / original.bbox_diagonal
        * 100,
    }


def run(decimation_ratio: float = 0.25) -> None:
    print(f"Loading fixture mesh: {FIXTURE_MESH}")
    original_mesh = load_mesh(FIXTURE_MESH)
    original_stats = trimesh_stats(original_mesh)

    print(f"Running pymeshlab decimation with ratio={decimation_ratio}")
    decimated_mesh = decimate_with_pymeshlab(original_mesh, decimation_ratio)
    decimated_stats = trimesh_stats(decimated_mesh)

    drift = compare_stats(original_stats, decimated_stats)

    print("\nOriginal mesh:")
    for key, value in original_stats.to_dict().items():
        print(f"  {key}: {value}")

    print("\nDecimated mesh:")
    for key, value in decimated_stats.to_dict().items():
        print(f"  {key}: {value}")

    print("\nMetric deltas (% where applicable):")
    for key, value in drift.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    run()
