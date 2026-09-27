"""Synthetic CPU checks of the included Orbit-GOP core, without study data."""

import unittest

import numpy as np
import torch

from models.registry import create_model
from orbit_gop.functional_signature import extract_batch
from orbit_gop.functional_risk import midrank, q_func
from orbit_gop.inference import calibrate
from orbit_gop.metrics import aurc, ood_metrics


class CoreCpuTests(unittest.TestCase):
    def test_three_backbone_signatures_and_bounds(self):
        torch.set_num_threads(2)
        torch.manual_seed(17)
        image = torch.rand((1, 3, 32, 32))
        for backbone, channels in [('resnet18', 256), ('vgg16_bn', 512), ('wrn28_10', 640)]:
            with self.subTest(backbone=backbone):
                model = create_model(backbone).cpu().eval()
                output = extract_batch(model, image, backbone)
                self.assertEqual(tuple(output['signatures'].shape), (1, 3, channels))
                np.testing.assert_allclose(output['signatures'].sum(-1).detach().numpy(), 1.0, atol=1e-6)
                risk = float(output['d_gop_L3'][0])
                self.assertTrue(np.isfinite(risk) and 0 <= risk <= 1)
                self.assertEqual(tuple(output['prediction'].shape), (1,))
                del model

    def test_midrank_prediction_and_metric_semantics(self):
        self.assertEqual(float(midrank(np.array([0.1, 0.2, 0.2, 0.3]), np.array([0.2]))[0]), 0.5)
        calibration = {'prediction': np.arange(10), 'd_gop_L3': np.arange(10) / 10}
        target = {'prediction': np.array([3]), 'd_gop_L3': np.array([0.3])}
        self.assertEqual(float(q_func(calibration, target, 'L3')[0]), 0.5)
        evidence = np.array([[0., 0., 0., 1., 0., 0., 0., 0., 0., 0.]])
        output = calibrate(evidence, np.array([0.5]), 2.0, np.array([3]))
        self.assertEqual(int(output['prediction'][0]), 3)
        self.assertEqual(float(output['numeric_prediction_changes'][0]), 0.0)
        self.assertEqual(ood_metrics(np.array([0.0, 0.2]), np.array([0.8, 1.0])),
                         {'AUROC': 1.0, 'FPR95': 0.0})
        self.assertEqual(aurc(np.array([0.1, 0.2]), np.array([True, False]), np.array([0, 1])), 0.25)


if __name__ == '__main__':
    unittest.main()
