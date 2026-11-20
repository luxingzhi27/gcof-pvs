from collections import Counter
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from gcof_pvs.data.pose_csr import PoseCSRDataset
from gcof_pvs.data.relations import build_proxy_relation_graph
from gcof_pvs.data.synthetic.build import generate_scene_dataset
from gcof_pvs.data.synthetic.catalog import generate_synthetic_scene_manifest
from gcof_pvs.evaluation.culling import candidate_normalized_occlusion_recall
from gcof_pvs.model.network import GCOFPVSV5
from gcof_pvs.protocol import MODEL_VARIANTS
from gcof_pvs.training.objective import pose_balanced_bce, shared_boundary_risks
from gcof_pvs.training.scenes import build_training_context
from gcof_pvs.training.schedule import balanced_step_schedule


class ProtocolTest(unittest.TestCase):
    def test_loso_only_resolves_source_scenes_and_uses_data_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            specs, policy = build_training_context(
                protocol="loso", held_out_scene="hkust", data_root=root, require_files=False
            )
            self.assertEqual(Counter(s.source_kind for s in specs), {"real": 4, "synthetic": 96})
            self.assertNotIn("hkust", {s.scene_id for s in specs})
            self.assertTrue(all(s.pose_dataset.is_relative_to(root) for s in specs))
            self.assertFalse(policy.to_manifest()["allowHeldOutVisibilityLabels"])

    def test_schedule_keeps_two_real_updates_per_synthetic_update(self):
        args = (["real_a", "real_b"], ["synth_a", "synth_b"])
        steps = list(balanced_step_schedule(*args, real_updates_per_scene=4, seed=11))
        self.assertEqual(steps, list(balanced_step_schedule(*args, real_updates_per_scene=4, seed=11)))
        self.assertEqual(Counter(s.source_kind for s in steps), {"real": 8, "synthetic": 4})

    def test_synthetic_pose_csr_is_evaluable(self):
        catalog = generate_synthetic_scene_manifest()
        self.assertEqual(catalog["splitCounts"], {"train": 96, "validation": 12, "diagnostic": 12})
        scene = dict(catalog["scenes"][0], unitCount=64)
        with tempfile.TemporaryDirectory() as tmp:
            generate_scene_dataset(scene, tmp, views_per_scene=8, render_width=24,
                                   render_height=16, allow_small_scene=True)
            dataset = PoseCSRDataset(Path(tmp) / scene["sceneId"] / "pose_csr", scene["unitCount"])
            self.assertEqual(len(dataset.poses), 8)
            for idx in range(len(dataset.poses)):
                ids, weights = dataset.visible_slice(idx)
                self.assertTrue(np.isin(ids, dataset.candidate_slice(idx)).all())
                self.assertTrue(np.isfinite(weights).all())


class ModelTest(unittest.TestCase):
    def test_representation_variants_query_with_expected_shapes(self):
        aabbs = np.array([[[0, 0, 0], [1, 1, 1]], [[0, 0, 2], [1, 1, 3]]], dtype=np.float32)
        graph = build_proxy_relation_graph(aabbs)
        relation = {
            "metadata": graph.manifest,
            "source_ids": graph.source_ids,
            "edge_features": graph.edge_features,
            "valid_mask": graph.valid_mask,
        }
        geometry = torch.zeros(2, 32)
        query = torch.zeros(2, 16)
        for variant in MODEL_VARIANTS:
            with self.subTest(variant=variant):
                model = GCOFPVSV5(variant)
                inputs = None if variant == "GEOMETRY_FIELD" else relation
                compiled = model.compile_outputs(geometry, inputs)
                key, shape = ("generic_latent", (2, 28)) if variant == "GENERIC_RELATION_28" else ("field", (2, 4, 7))
                self.assertEqual(tuple(compiled[key].shape), shape)
                logits = model.query(geometry, query, field_stats=torch.zeros(2, 4),
                                     **{key: compiled[key]})
                self.assertEqual(tuple(logits.shape), (2, 1))
                self.assertTrue(torch.isfinite(logits).all())

    def test_objectives_and_candidate_normalization(self):
        logits = torch.zeros(2, requires_grad=True)
        labels = torch.tensor([1.0, 0.0])
        risks = shared_boundary_risks(logits, labels, labels, 1.0, 1.0, 1.0, rho=0.1)
        self.assertAlmostEqual(risks.extra.item(), 1.0)
        self.assertAlmostEqual(risks.safety.item(), 1.0)
        self.assertAlmostEqual(pose_balanced_bce(logits, labels, [0, 2], 1).item(), math.log(2))
        (risks.extra + risks.safety).backward()
        self.assertLess(logits.grad[0].item(), 0)
        self.assertGreater(logits.grad[1].item(), 0)
        self.assertAlmostEqual(
            candidate_normalized_occlusion_recall([1, 18], [0, 2], [2, 100]),
            (0.5 + 0.18) / (0.5 + 0.20),
        )
