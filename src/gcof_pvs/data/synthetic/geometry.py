"""Deterministic scene layouts and primitive geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from gcof_pvs.data.surface import SurfaceUnitInput


_PRIMITIVE_TYPES = ("box", "cylinder", "beam", "panel")


def _normalize(value: np.ndarray, fallback: Sequence[float]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    length = float(np.linalg.norm(array))
    if length <= 1.0e-12:
        return np.asarray(fallback, dtype=np.float64)
    return array / length


def _rotation_from_euler(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Return a row-vector-compatible local-to-world rotation."""

    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.asarray([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    ry = np.asarray([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return rz @ ry @ rx


def _box_mesh(dimensions: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    half = np.asarray(dimensions, dtype=np.float64) * 0.5
    vertices = np.asarray(
        [
            (-half[0], -half[1], -half[2]),
            (half[0], -half[1], -half[2]),
            (half[0], half[1], -half[2]),
            (-half[0], half[1], -half[2]),
            (-half[0], -half[1], half[2]),
            (half[0], -half[1], half[2]),
            (half[0], half[1], half[2]),
            (-half[0], half[1], half[2]),
        ],
        dtype=np.float64,
    )
    triangles = np.asarray(
        [
            (0, 1, 2),
            (0, 2, 3),
            (4, 6, 5),
            (4, 7, 6),
            (0, 4, 5),
            (0, 5, 1),
            (3, 2, 6),
            (3, 6, 7),
            (0, 3, 7),
            (0, 7, 4),
            (1, 5, 6),
            (1, 6, 2),
        ],
        dtype=np.int64,
    )
    return vertices, triangles


def _cylinder_mesh(radius: float, height: float, segments: int = 16) -> tuple[np.ndarray, np.ndarray]:
    angles = np.arange(segments, dtype=np.float64) * (2.0 * math.pi / segments)
    ring = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])
    vertices = np.vstack(
        [
            np.column_stack([ring[:, 0], np.full(segments, -height * 0.5), ring[:, 1]]),
            np.column_stack([ring[:, 0], np.full(segments, height * 0.5), ring[:, 1]]),
            np.asarray([[0.0, -height * 0.5, 0.0], [0.0, height * 0.5, 0.0]]),
        ]
    )
    bottom_center = 2 * segments
    top_center = bottom_center + 1
    triangles: list[tuple[int, int, int]] = []
    for index in range(segments):
        nxt = (index + 1) % segments
        triangles.extend(
            [
                (index, nxt, segments + nxt),
                (index, segments + nxt, segments + index),
                (bottom_center, nxt, index),
                (top_center, segments + index, segments + nxt),
            ]
        )
    return vertices.astype(np.float64), np.asarray(triangles, dtype=np.int64)


@dataclass(frozen=True)
class PrimitiveUnit:
    """One actual renderable unit with analytic and triangle geometry."""

    unit_id: int
    primitive_type: str
    center: np.ndarray
    rotation: np.ndarray
    dimensions: np.ndarray
    local_vertices: np.ndarray
    local_triangles: np.ndarray
    resource_id: int
    world_aabb: np.ndarray

    def surface_input(self) -> SurfaceUnitInput:
        transform = np.eye(4, dtype=np.float64)
        transform[:3, :3] = self.rotation
        transform[:3, 3] = self.center
        return SurfaceUnitInput(
            self.unit_id,
            self.local_vertices,
            self.local_triangles,
            transform,
        )

    def mapping(self) -> dict[str, Any]:
        return {
            "unitId": self.unit_id,
            "primitiveType": self.primitive_type,
            "center": self.center.astype(float).tolist(),
            "rotation": self.rotation.astype(float).tolist(),
            "dimensions": self.dimensions.astype(float).tolist(),
            "resourceId": self.resource_id,
            "bounds": {
                "min": self.world_aabb[0].astype(float).tolist(),
                "max": self.world_aabb[1].astype(float).tolist(),
            },
        }


@dataclass
class PrimitiveScene:
    scene_id: str
    units: tuple[PrimitiveUnit, ...]
    bounds: np.ndarray
    _grid: Any | None = None

    @property
    def unit_count(self) -> int:
        return len(self.units)

    @property
    def unit_ids(self) -> np.ndarray:
        return np.asarray([unit.unit_id for unit in self.units], dtype=np.int64)

    @property
    def primitive_types(self) -> np.ndarray:
        return np.asarray([unit.primitive_type for unit in self.units], dtype="U8")

    @property
    def centers(self) -> np.ndarray:
        return np.asarray([unit.center for unit in self.units], dtype=np.float64)

    @property
    def rotations(self) -> np.ndarray:
        return np.asarray([unit.rotation for unit in self.units], dtype=np.float64)

    @property
    def dimensions(self) -> np.ndarray:
        return np.asarray([unit.dimensions for unit in self.units], dtype=np.float64)

    @property
    def aabbs(self) -> np.ndarray:
        return np.asarray([unit.world_aabb for unit in self.units], dtype=np.float64)

    def surface_inputs(self) -> list[SurfaceUnitInput]:
        return [unit.surface_input() for unit in self.units]


def _primitive_mesh(primitive_type: str, dimensions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if primitive_type == "cylinder":
        return _cylinder_mesh(float(dimensions[0]) * 0.5, float(dimensions[1]))
    return _box_mesh(dimensions)


def _family_position(
    family_index: int,
    unit_index: int,
    unit_count: int,
    *,
    rng: np.random.Generator,
    channel_width: float,
    layers: int,
    scale: float,
) -> tuple[np.ndarray, float, float, float]:
    """Place units on deterministic structural scaffolds, not random AABBs."""

    index = int(unit_index)
    jitter = rng.uniform(-0.18, 0.18, size=3) * scale
    if family_index == 0:
        span = max(4, int(math.ceil(math.sqrt(unit_count / max(1, layers)))))
        segment = index // 4
        lane = index % 4
        x = (segment % span - (span - 1) * 0.5) * channel_width
        z = (segment // span - max(1, layers - 1) * 0.5) * channel_width * 1.4
        y = (lane - 1.5) * max(1.4, channel_width * 0.22)
        return np.asarray([x, y, z]) + jitter, float(index % 3) * 0.2, 0.0, 0.0
    if family_index == 1:
        side = max(4, int(math.ceil(math.sqrt(unit_count / max(1, layers)))))
        gx = index % side
        gz = (index // side) % side
        level = (index // (side * side)) % max(1, layers)
        spacing = max(3.0, channel_width * 0.8)
        position = np.asarray(
            [(gx - (side - 1) * 0.5) * spacing, level * max(3.0, scale * 2.8), (gz - (side - 1) * 0.5) * spacing]
        )
        return position + jitter, float((gx + gz) % 4) * 0.15, 0.0, 0.0
    if family_index == 2:
        rows = max(2, int(math.ceil(math.sqrt(unit_count / 2.0))))
        row = index // 2
        side = -1.0 if index % 2 == 0 else 1.0
        z = (row % rows - (rows - 1) * 0.5) * max(4.0, channel_width)
        y = ((row // rows) % max(1, layers)) * max(5.0, scale * 3.5)
        x = side * max(8.0, channel_width * 1.8)
        return np.asarray([x, y, z]) + jitter, side * 0.08, 0.0, 0.0
    if family_index == 3:
        grid = max(3, int(math.ceil(unit_count ** (1.0 / 3.0))))
        gx = index % grid
        gz = (index // grid) % grid
        level = (index // (grid * grid)) % max(1, layers)
        position = np.asarray(
            [
                (gx - (grid - 1) * 0.5) * max(2.0, channel_width * 0.5),
                level * max(2.5, scale * 1.5),
                (gz - (grid - 1) * 0.5) * max(2.0, channel_width * 0.5),
            ]
        )
        return position + jitter, float(index % 8) * math.pi / 8.0, 0.0, 0.0
    # The clutter family uses a 3-D lattice with a deterministic low-discrepancy
    # offset.  Repeated lattice locations receive repeated primitive shapes,
    # while the jitter and type cycle retain non-repeated units.
    grid = max(4, int(math.ceil(unit_count ** (1.0 / 3.0))))
    gx = index % grid
    gz = (index // grid) % grid
    level = (index // (grid * grid)) % max(1, layers)
    position = np.asarray(
        [
            (gx - (grid - 1) * 0.5) * max(2.0, scale * 1.8),
            level * max(2.0, scale * 1.4),
            (gz - (grid - 1) * 0.5) * max(2.0, scale * 1.8),
        ]
    )
    return position + jitter, float((index * 13) % 16) * math.pi / 16.0, 0.0, 0.0


def build_primitive_scene(scene: Mapping[str, Any], *, allow_small: bool = False) -> PrimitiveScene:
    """Materialize all renderable units for one catalog entry."""

    required = {"sceneId", "generatorSeed", "unitCount", "geometryRecipe", "randomization"}
    missing = sorted(required - set(scene))
    if missing:
        raise ValueError(f"scene is missing geometry fields: {', '.join(missing)}")
    count = int(scene["unitCount"])
    if (not allow_small and not 256 <= count <= 4096) or count <= 0:
        raise ValueError("synthetic scenes must contain 256..4096 units")
    recipe = scene["geometryRecipe"]
    randomization = scene["randomization"]
    family_index = int(recipe["familyIndex"])
    if family_index not in range(5):
        raise ValueError("geometryRecipe.familyIndex must be in [0,4]")
    seed = int(scene["generatorSeed"])
    rng = np.random.default_rng(seed)
    scale_range = randomization.get("scaleRange", [0.8, 2.0])
    base_scale = float(rng.uniform(float(scale_range[0]), float(scale_range[1])))
    channel_width = float(randomization.get("channelWidth", 6.0))
    layers = int(randomization.get("layers", 3))
    size_distribution = float(randomization.get("unitSizeDistribution", 0.5))
    repeat_rate = float(randomization.get("repeatRate", 0.5))
    type_patterns = (
        ("panel", "box", "beam", "box", "cylinder"),
        ("cylinder", "beam", "box", "panel", "box"),
        ("box", "panel", "box", "beam", "cylinder"),
        ("cylinder", "beam", "panel", "box", "cylinder"),
        ("box", "cylinder", "beam", "panel", "box", "cylinder"),
    )
    units: list[PrimitiveUnit] = []
    for unit_id in range(count):
        if unit_id < 4:
            primitive_type = _PRIMITIVE_TYPES[unit_id]
        else:
            primitive_type = type_patterns[family_index][unit_id % len(type_patterns[family_index])]
        center, yaw, pitch, roll = _family_position(
            family_index,
            unit_id,
            count,
            rng=rng,
            channel_width=channel_width,
            layers=layers,
            scale=base_scale,
        )
        size_factor = float(rng.uniform(0.65, 1.45)) ** (0.5 + size_distribution)
        if primitive_type == "cylinder":
            radius = max(0.12, base_scale * 0.22 * size_factor)
            height = max(0.35, base_scale * rng.uniform(1.2, 3.8) * size_factor)
            dimensions = np.asarray([radius * 2.0, height, radius * 2.0])
        elif primitive_type == "beam":
            dimensions = np.asarray(
                [
                    max(0.5, base_scale * rng.uniform(2.0, 7.0) * size_factor),
                    max(0.22, base_scale * rng.uniform(0.18, 0.55)),
                    max(0.22, base_scale * rng.uniform(0.18, 0.55)),
                ]
            )
        elif primitive_type == "panel":
            dimensions = np.asarray(
                [
                    max(0.8, base_scale * rng.uniform(1.5, 5.5) * size_factor),
                    max(0.8, base_scale * rng.uniform(1.2, 4.0) * size_factor),
                    max(0.08, base_scale * rng.uniform(0.08, 0.24)),
                ]
            )
        else:
            dimensions = np.asarray(
                [
                    max(0.5, base_scale * rng.uniform(0.55, 2.4) * size_factor),
                    max(0.5, base_scale * rng.uniform(0.45, 2.4) * size_factor),
                    max(0.5, base_scale * rng.uniform(0.55, 2.4) * size_factor),
                ]
            )
        # Structural units sit on or around their scaffold rather than sharing
        # a fabricated AABB.  Only a small deterministic vertical adjustment is
        # applied; the mesh remains the source of the final bounds.
        if family_index in (0, 1, 3, 4):
            center[1] += float(dimensions[1]) * 0.5
        rotation = _rotation_from_euler(yaw, pitch, roll)
        local_vertices, local_triangles = _primitive_mesh(primitive_type, dimensions)
        world_vertices = local_vertices @ rotation.T + center[None, :]
        world_aabb = np.stack([world_vertices.min(axis=0), world_vertices.max(axis=0)], axis=0)
        # Repetition is represented by a shared resource ID, while every unit
        # retains its own geometry and component ID.
        if repeat_rate > 0.0 and unit_id >= 4:
            repeat_group = max(1, int(round(1.0 + repeat_rate * 7.0)))
            resource_id = int(unit_id % repeat_group)
        else:
            resource_id = unit_id
        units.append(
            PrimitiveUnit(
                unit_id,
                primitive_type,
                center.astype(np.float64),
                rotation.astype(np.float64),
                dimensions.astype(np.float64),
                local_vertices,
                local_triangles,
                resource_id,
                world_aabb.astype(np.float64),
            )
        )
    bounds = np.stack(
        [np.asarray([unit.world_aabb[0] for unit in units]).min(axis=0), np.asarray([unit.world_aabb[1] for unit in units]).max(axis=0)],
        axis=0,
    )
    return PrimitiveScene(str(scene["sceneId"]), tuple(units), bounds.astype(np.float64))
