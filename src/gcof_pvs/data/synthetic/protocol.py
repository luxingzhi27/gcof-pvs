"""Frozen procedural dataset defaults and schema identifiers."""

from pathlib import Path


EXPERIMENT_NAME = "pvs_v5_synthetic_procedural_primitives_v1"
DEFAULT_BASE_SEED = 20260918
DEFAULT_OUTPUT_ROOT = Path("data/synthetic")
DEFAULT_VIEWS_PER_SCENE = 64
DEFAULT_RENDER_WIDTH = 160
DEFAULT_RENDER_HEIGHT = 90
SUBPOSES_PER_VIEWCELL = 4
MODEL_FOV_Y_DEG = 66.0
RENDER_FOV_Y_DEG = 60.0
DEFAULT_NEAR = 0.05
DEFAULT_FAR = 1_000_000.0
SYNTHETIC_CAMERA_ORBIT_FRACTION = 0.20
DATASET_SCHEMA = "gcof-pvs-v5-synthetic-dataset-v1"
SCENE_OUTPUT_SCHEMA = "gcof-pvs-v5-synthetic-scene-output-v1"
COMPILED_SCHEMA = "gcof-pvs-v5-compiled-scene-v1"
