import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch and project dependencies are required')
class OverfitProbeTests(unittest.TestCase):
    def test_fresh_initialization_is_reproducible_and_does_not_load_source_weights(self):
        from unittest.mock import patch
        from scripts.probe_rpn_overfit import build_modules, seed

        with patch('scripts.probe_rpn_overfit.FeatureNet', lambda cfg: torch.nn.Linear(2, 2)), \
             patch('scripts.probe_rpn_overfit.RpnHead', lambda cfg, channels: torch.nn.Linear(2, 1)), \
             patch.object(torch.nn.Module, 'cuda', lambda module: module):
            state = {'feature_net.weight': torch.full((2, 2), 9.),
                     'feature_net.bias': torch.full((2,), 9.),
                     'rpn.weight': torch.full((1, 2), 9.), 'rpn.bias': torch.full((1,), 9.)}
            loaded = build_modules({}, state)
            self.assertTrue(all(torch.all(p == 9) for m in loaded for p in m.parameters()))
            seed(35202)
            first = build_modules({}, None)
            seed(35202)
            second = build_modules({}, None)
            for a, b in zip(first, second):
                for p, q in zip(a.parameters(), b.parameters()):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)
                    self.assertFalse(torch.all(p == 9))
            self.assertTrue(all(torch.all(t == 9) for t in state.values()))

    def test_local_probe_maps_anchor_sites_and_detects_relu_blocking(self):
        from unittest.mock import patch
        from scripts.probe_rpn_overfit import inspect_local_features
        from net.heads import RpnHead

        class Feature(torch.nn.Module):
            def __init__(self, sign):
                super().__init__()
                self.back2 = torch.nn.Sequential(torch.nn.Conv3d(1, 2, 1, bias=False),
                                                torch.nn.ReLU(inplace=True))
                torch.nn.init.constant_(self.back2[0].weight, sign)

            def forward(self, x):
                return [self.back2(x)], None

        cfg = dict(anchors=[[5, 5, 5], [10, 10, 10]], num_hard=2,
                   head_activation='leaky_relu')
        for sign in (-1., 1.):
            with self.subTest(sign=sign):
                feature = Feature(sign)
                head = RpnHead(cfg, 2)
                for p in head.parameters():
                    torch.nn.init.constant_(p, .1)
                labels = torch.tensor([[0., 0., 0., 1., 0., 0.]])
                sample = dict(image=torch.tensor([1., 2., 3.]).reshape(1, 1, 1, 1, 3),
                              targets=(labels, None, torch.ones_like(labels), None, None))
                before = {k: v.clone() for k, v in feature.state_dict().items()}
                with patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
                    result = inspect_local_features(feature, head, sample, cfg)[0]
                self.assertEqual(result['anchor_kind'], 1)
                self.assertEqual(result['feature_site_zyx'], [0, 0, 1])
                self.assertGreater(result['positive_logit_grad_after_relu_l2'], 0.)
                if sign < 0:
                    self.assertEqual(result['positive_feature_zero_fraction'], 1.)
                    self.assertEqual(result['positive_logit_grad_before_relu_l2'], 0.)
                    self.assertEqual(result['same_kind_hard_negative_near_identical_count'], 2)
                    # Pointwise convolutions on different grid sizes can round differently in FP32.
                    self.assertAlmostEqual(result['positive_probability'], result['zero_input_probability'],
                                           delta=torch.finfo(torch.float32).eps)
                else:
                    self.assertGreater(result['positive_logit_grad_before_relu_l2'], 0.)
                    self.assertEqual(result['same_kind_hard_negative_near_identical_count'], 0)
                for name, value in feature.state_dict().items():
                    torch.testing.assert_close(before[name], value, rtol=0, atol=0)
                self.assertFalse(feature.back2[-1]._forward_pre_hooks)

    def test_metrics_distinguish_sampled_and_hard_negatives(self):
        from scripts.probe_rpn_overfit import predictions

        class Head(torch.nn.Module):
            def forward(self, feature):
                # Last anchor is ignored even though its score is the highest.
                p = torch.tensor([.4, .9, .2, .99])
                return torch.logit(p).reshape(1, -1, 1), torch.zeros(1, 4, 6)

        labels = torch.tensor([[1., 0., 0., 0.]])
        weights = torch.tensor([[1., 1., 1., 0.]])
        targets = (labels, torch.zeros_like(labels), weights,
                   torch.zeros(1, 4, 6), labels.clone())
        loss, metrics = predictions(Head(), torch.ones(1, 2, 1, 1, 4), targets, {'num_hard': 1})
        self.assertAlmostEqual(metrics['positive_probability'], .4, places=6)
        self.assertAlmostEqual(metrics['sampled_negative_probability'], .55, places=6)
        self.assertAlmostEqual(metrics['hard_negative_probability'], .9, places=6)
        self.assertEqual(metrics['sampled_negatives_above_positive'], 1)
        self.assertEqual(metrics['sampled_negative_count'], 2)
        self.assertAlmostEqual(float(loss), -.5 * (torch.log(torch.tensor(.4))
                                                  + torch.log(torch.tensor(.1))).item(), places=6)

    def test_diagnostic_preserves_gradient_flow_and_targets(self):
        from scripts.probe_rpn_overfit import predictions, grad_norm
        from net.heads import RpnHead
        cfg = dict(anchors=[[5, 5, 5]], num_hard=1, head_activation='leaky_relu')
        head = RpnHead(cfg, 2)
        feature = torch.randn(1, 2, 1, 1, 4, requires_grad=True)
        labels = torch.tensor([[1., 0., 0., 0.]])
        targets = (labels, torch.zeros_like(labels), torch.ones_like(labels),
                   torch.zeros(1, 4, 6), labels.clone())
        original = [t.clone() for t in targets]
        loss, _ = predictions(head, feature, targets, cfg)
        loss.backward()
        self.assertGreater(grad_norm(head), 0.)
        self.assertGreater(float(feature.grad.norm()), 0.)
        for before, after in zip(original, targets):
            torch.testing.assert_close(before, after, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
