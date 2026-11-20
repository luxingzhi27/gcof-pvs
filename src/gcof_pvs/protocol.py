"""Stable variants and artifact contracts shared by training and evaluation."""

MODEL_VARIANTS = ("FULL", "GEOMETRY_FIELD", "GENERIC_RELATION_28")
TRAINING_VARIANTS = (*MODEL_VARIANTS, "PBCE_OBJECTIVE")
ROBUST_BOUNDARY_OBJECTIVE = "ROBUST_BOUNDARY"
RUN_SCHEMA = "gcof-pvs-v5-formal-training-run-v2"
CHECKPOINT_SCHEMA = "gcof-pvs-v5-training-checkpoint-v2"
