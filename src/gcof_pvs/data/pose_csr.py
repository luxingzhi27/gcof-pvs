from __future__ import annotations

import json
from pathlib import Path

import numpy as np


DIRECTIONAL_POSE_DTYPE = np.dtype(
    {
        "names": [
            "camera_norm",
            "camera_world",
            "camera_forward",
            "camera_view",
            "split",
            "category",
            "reserved0",
            "reserved1",
        ],
        "formats": [
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (3,)),
            ("<f4", (2,)),
            "u1",
            "u1",
            "<u2",
            "<u4",
        ],
        "offsets": [0, 12, 24, 36, 44, 45, 46, 48],
        "itemsize": 64,
    }
)


class PoseCSRSplit:
    """Read-only pose indices for one declared split."""

    def __init__(self, dataset: "PoseCSRDataset", split_name: str):
        self.pose_indices = np.flatnonzero(
            dataset.poses["split"] == dataset.split_ids[split_name]
        ).astype(np.int64, copy=False)


class PoseCSRDataset:
    """Memory-mapped directional view-cell poses, candidates and GT weights."""

    def __init__(
        self,
        dataset_dir: str | Path,
        num_instances: int,
    ):
        self.dataset_dir = Path(dataset_dir)
        self.meta = self._read_json("dataset_meta.json")
        self.split_ids = self.meta["splitIds"]
        self.num_instances = int(num_instances)
        if int(self.meta["poseStrideBytes"]) != DIRECTIONAL_POSE_DTYPE.itemsize:
            raise ValueError("V5 requires directional poses")
        self.pose_dtype = DIRECTIONAL_POSE_DTYPE
        self.poses = np.memmap(self.dataset_dir / "poses.bin", dtype=self.pose_dtype, mode="r")
        self.visible_offsets = np.memmap(self.dataset_dir / "visible_offsets.bin", dtype=np.uint64, mode="r")
        visible_ids_path = self.dataset_dir / "visible_ids.bin"
        if visible_ids_path.stat().st_size == 0:
            self.visible_ids = np.zeros((0,), dtype=np.uint32)
        else:
            self.visible_ids = np.memmap(visible_ids_path, dtype=np.uint32, mode="r")
        weights_path = self.dataset_dir / "visible_weights.bin"
        self.visible_weight_source = "visible_weights.bin"
        self.visible_weight_semantics = self.meta.get(
            "visibleWeightSemantics",
            "rvcServer component_weights, not pixel count",
        )
        if weights_path.exists() and weights_path.stat().st_size == 0:
            self.visible_weights = np.zeros((0,), dtype=np.float32)
        elif weights_path.exists():
            weight_dtype_name = str(self.meta.get("visibleWeightDtype", "float32")).lower()
            weight_dtype = np.float32 if weight_dtype_name in {"float", "float32", "f4", "<f4"} else np.uint32
            self.visible_weights = np.memmap(weights_path, dtype=weight_dtype, mode="r")
        else:
            raise FileNotFoundError(weights_path)
        self.candidate_offsets = np.memmap(self.dataset_dir / "candidate_offsets.bin", dtype=np.uint64, mode="r")
        candidate_path = self.dataset_dir / "candidate_ids.bin"
        if candidate_path.stat().st_size == 0:
            self.candidate_ids = np.zeros((0,), dtype=np.uint32)
        else:
            self.candidate_ids = np.memmap(candidate_path, dtype=np.uint32, mode="r")
        self.visible_counts = np.diff(self.visible_offsets).astype(np.int64, copy=False)
        self.candidate_counts = np.diff(self.candidate_offsets).astype(np.int64, copy=False)
        files = self.meta.get("files", {})
        query_center_name = files.get("queryCenterWorld", "query_center_world.bin")
        candidate_camera_name = files.get("candidateCameraWorld", "candidate_camera_world.bin")
        radius_name = files.get("viewcellRadiusM", "viewcell_radius_m.bin")
        query_center_path = self.dataset_dir / query_center_name
        candidate_camera_path = self.dataset_dir / candidate_camera_name
        radius_path = self.dataset_dir / radius_name
        self.query_centers_world = None
        self.candidate_cameras_world = None
        self.viewcell_radii_m = None
        if query_center_path.exists():
            values = np.memmap(query_center_path, dtype=np.float32, mode="r")
            if values.size != self.poses.size * 3:
                raise ValueError("query_center_world.bin does not match pose count")
            self.query_centers_world = values.reshape(-1, 3)
        if candidate_camera_path.exists():
            values = np.memmap(candidate_camera_path, dtype=np.float32, mode="r")
            if values.size != self.poses.size * 3:
                raise ValueError("candidate_camera_world.bin does not match pose count")
            values = values.reshape(-1, 3)
            pose_camera = np.asarray(self.poses["camera_world"], dtype=np.float32)
            if not np.allclose(values, pose_camera, rtol=0.0, atol=1e-5):
                raise ValueError("candidate_camera_world.bin disagrees with poses.bin")
            self.candidate_cameras_world = values
        if radius_path.exists():
            values = np.memmap(radius_path, dtype=np.float32, mode="r")
            if values.size != self.poses.size:
                raise ValueError("viewcell_radius_m.bin does not match pose count")
            if np.any(~np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("viewcell_radius_m.bin contains invalid radii")
            self.viewcell_radii_m = values
        if (self.query_centers_world is None) != (self.viewcell_radii_m is None):
            raise ValueError("query center and view-cell radius files must be provided together")

    def _read_json(self, name: str) -> dict:
        return json.loads((self.dataset_dir / name).read_text(encoding="utf-8"))

    def split(self, split_name: str) -> PoseCSRSplit:
        return PoseCSRSplit(self, split_name)

    def visible_slice(self, pose_index: int) -> tuple[np.ndarray, np.ndarray]:
        start = int(self.visible_offsets[pose_index])
        end = int(self.visible_offsets[pose_index + 1])
        return self.visible_ids[start:end], self.visible_weights[start:end]

    def candidate_slice(self, pose_index: int) -> np.ndarray:
        start = int(self.candidate_offsets[pose_index])
        end = int(self.candidate_offsets[pose_index + 1])
        return self.candidate_ids[start:end]

    def camera_view(self, pose_index: int) -> np.ndarray:
        pose = self.poses[pose_index]
        return np.concatenate([pose["camera_forward"], pose["camera_view"]]).astype(np.float32, copy=False)

    def query_center_world(self, pose_index: int, *, required: bool = False) -> np.ndarray:
        """Return the view-cell query center, distinct from the candidate camera."""
        if self.query_centers_world is None:
            if required:
                raise ValueError(
                    "this model requires query_center_world.bin; rebuild the directional view-cell PoseCSR"
                )
            return np.asarray(self.poses["camera_world"][pose_index], dtype=np.float32)
        return np.asarray(self.query_centers_world[pose_index], dtype=np.float32)

    def candidate_camera_world(self, pose_index: int, *, required: bool = False) -> np.ndarray:
        """Return the back-camera center used only to define candidates."""
        if self.candidate_cameras_world is None:
            if required:
                raise ValueError(
                    "this model requires candidate_camera_world.bin; rebuild the directional view-cell PoseCSR"
                )
            return np.asarray(self.poses["camera_world"][pose_index], dtype=np.float32)
        return np.asarray(self.candidate_cameras_world[pose_index], dtype=np.float32)

    def viewcell_radius_m(self, pose_index: int, *, required: bool = False) -> float:
        if self.viewcell_radii_m is None:
            if required:
                raise ValueError(
                    "this model requires viewcell_radius_m.bin; rebuild the directional view-cell PoseCSR"
                )
            return 0.0
        return float(self.viewcell_radii_m[pose_index])


def _normalize(v: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return fallback.astype(np.float32, copy=True)
    return (v / n).astype(np.float32, copy=False)


def frustum_candidate_ids_for_pose(
    camera_world: np.ndarray,
    forward: np.ndarray,
    tan_x: float,
    tan_y: float,
    world_aabbs: np.ndarray,
    near: float = 0.05,
) -> np.ndarray:
    """训练/评估阶段的保守 AABB-vs-camera-frustum 候选测试。

    它近似前端 Three.js Frustum.intersectsBox 的作用：保留投影区间可能和水平/垂直
    FOV 重叠的 AABB。这里故意偏保守，宁愿多保留候选，也不要漏掉 GT 正样本。
    漏正样本会直接破坏 set-level 训练，多几个负样本只会增加一点训练压力。
    """
    if world_aabbs.size == 0:
        return np.zeros((0,), dtype=np.uint32)
    f = _normalize(np.asarray(forward, dtype=np.float32), np.asarray([0.0, 0.0, -1.0], dtype=np.float32))
    up_ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    if abs(float(np.dot(f, up_ref))) > 0.98:
        up_ref = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    right = _normalize(np.cross(f, up_ref), np.asarray([1.0, 0.0, 0.0], dtype=np.float32))
    up = _normalize(np.cross(right, f), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))

    mins = world_aabbs[:, :3]
    maxs = world_aabbs[:, 3:]
    centers = (mins + maxs) * 0.5
    extents = np.maximum((maxs - mins) * 0.5, 0.0)
    d = centers - np.asarray(camera_world, dtype=np.float32)[None, :]
    z = d @ f
    x = d @ right
    y = d @ up
    rz = extents @ np.abs(f)
    rx = extents @ np.abs(right)
    ry = extents @ np.abs(up)
    front = z + rz >= float(near)
    x_inside = (np.abs(x) - rx) <= ((z + rz) * max(float(tan_x), 1e-4))
    y_inside = (np.abs(y) - ry) <= ((z + rz) * max(float(tan_y), 1e-4))
    return np.flatnonzero(front & x_inside & y_inside).astype(np.uint32)
