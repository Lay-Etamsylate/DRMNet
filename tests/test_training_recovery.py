import copy
import unittest

from net.training_setup import (resolve_model_for_training, resolve_training_settings,
                                load_training_weights)

try:
    import torch
except ImportError:
    torch = None

DEFAULTS = dict(epochs=80, epoch_rcnn=21, optimizer='SGD', init_lr=.01,
                momentum=.9, weight_decay=.0001)


class RecoveryConfigTests(unittest.TestCase):
    def test_resume_restores_recovery_schedule(self):
        saved = dict(DEFAULTS, epoch_rcnn=6, init_lr=.0001, max_grad_norm=5.)
        cfg = resolve_training_settings(DEFAULTS, {'training_settings': saved}, epochs=None, init_lr=None)
        self.assertEqual(cfg, saved)
        self.assertEqual(resolve_training_settings(DEFAULTS, {'training_settings': saved}, epochs=100)['epochs'], 100)

    def test_legacy_defaults_and_invalid_options(self):
        self.assertEqual(resolve_training_settings(DEFAULTS)['max_grad_norm'], 0.)
        for options in ({'epochs': 0}, {'init_lr': 0}, {'max_grad_norm': -1}, {'init_lr': float('nan')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                resolve_training_settings(DEFAULTS, **options)

    def test_resume_cannot_silently_change_activation(self):
        checkpoint = {'model_config': {'model_variant': 'fusion'}}
        with self.assertRaisesRegex(ValueError, 'Cannot change'):
            resolve_model_for_training({}, checkpoint, head_activation='leaky_relu', resuming=True)
        cfg = resolve_model_for_training({}, checkpoint, head_activation='leaky_relu', reset_heads=True)
        self.assertEqual(cfg['head_activation'], 'leaky_relu')
        self.assertEqual(cfg['model_variant'], 'fusion')
        self.assertNotIn('head_activation', checkpoint['model_config'])

    def test_reset_requires_weights_only_source(self):
        with self.assertRaises(ValueError):
            resolve_model_for_training({}, reset_heads=True)
        with self.assertRaises(ValueError):
            resolve_model_for_training({}, {}, resuming=True, reset_heads=True)

    def test_saved_leaky_configuration_restores(self):
        source = {'model_config': {'model_variant': 'fusion', 'head_activation': 'leaky_relu',
                                   'rpn_train_pre_topk': 2000, 'rpn_train_post_topk': 300}}
        cfg = resolve_model_for_training({}, source, resuming=True)
        self.assertEqual(cfg['head_activation'], 'leaky_relu')
        self.assertEqual(cfg['rpn_train_pre_topk'], 2000)


@unittest.skipIf(torch is None, 'PyTorch is required')
class RecoveryWeightTests(unittest.TestCase):
    def setUp(self):
        from net.heads import RpnHead, RcnnHead
        torch.manual_seed(35202)
        self.cfg = dict(anchors=[[5, 5, 5]], num_class=2, rcnn_crop_size=(1, 1, 1),
                        head_activation='leaky_relu')
        self.model = torch.nn.Module()
        self.model.feature_net = torch.nn.Linear(2, 3)
        self.model.rcnn_crop = torch.nn.Linear(3, 2)
        self.model.rpn = RpnHead(self.cfg, in_channels=1)
        self.model.rcnn_head = RcnnHead(self.cfg, in_channels=1)
        with torch.no_grad():
            for p in self.model.parameters():
                p.fill_(2.)
        self.source = {'epoch': 80, 'state_dict': copy.deepcopy(self.model.state_dict())}

    def test_reset_preserves_other_weights_and_source(self):
        load_training_weights(self.model, self.source, self.cfg, reset_heads=True)
        for key, tensor in self.model.state_dict().items():
            if key.startswith(('rpn.', 'rcnn_head.')):
                self.assertFalse(torch.equal(tensor, self.source['state_dict'][key]))
            else:
                torch.testing.assert_close(tensor, self.source['state_dict'][key], rtol=0, atol=0)
        self.assertTrue(all(torch.all(tensor == 2) for tensor in self.source['state_dict'].values()))
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.0001, momentum=.9)
        self.assertFalse(optimizer.state)

    def test_plain_load_is_exact(self):
        load_training_weights(self.model, self.source, self.cfg)
        for key, tensor in self.model.state_dict().items():
            torch.testing.assert_close(tensor, self.source['state_dict'][key], rtol=0, atol=0)

    def test_reset_heads_have_finite_losses_and_hidden_gradients(self):
        load_training_weights(self.model, self.source, self.cfg, reset_heads=True)
        logits, deltas = self.model.rpn(torch.ones(1, 1, 2, 2, 2))
        rc_logits, rc_deltas = self.model.rcnn_head(torch.randn(4, 1, 1, 1, 1))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        loss = loss + torch.nn.functional.cross_entropy(rc_logits, torch.tensor([0, 1, 0, 1]))
        loss = loss + deltas.square().mean() + rc_deltas.square().mean()
        self.assertTrue(torch.isfinite(loss))
        self.assertLess(float(loss), 10.)
        loss.backward()
        for p in (self.model.rpn.conv[0].weight, self.model.rcnn_head.fc1.weight, self.model.rcnn_head.fc2.weight):
            self.assertGreater(float(p.grad.abs().sum()), 0.)
            self.assertTrue(torch.isfinite(p.grad).all())

    def test_proposal_caps_do_not_change_full_scan_eval(self):
        import numpy as np
        from unittest.mock import patch
        from net.layer.rpn_nms import rpn_nms
        cfg = dict(rpn_train_nms_pre_score_threshold=0., rpn_train_nms_overlap_threshold=.1,
                   rpn_test_nms_pre_score_threshold=0., rpn_test_nms_overlap_threshold=.1,
                   rpn_train_pre_topk=3, rpn_train_post_topk=2, box_reg_weight=[1.] * 6)
        windows = np.array([[10 * i, 10 * i, 10 * i, 2, 2, 2] for i in range(1, 6)], np.float32)
        inputs = torch.zeros(1, 1, 64, 64, 64)
        logits = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)
        deltas = torch.zeros(1, 5, 6)
        with patch.object(torch.Tensor, 'cuda', lambda x, *a, **kw: x):
            for mode in ('train', 'valid', 'eval', 'test'):
                result = rpn_nms(cfg, mode, inputs, windows, logits, deltas)
                self.assertEqual(len(result), 2 if mode in ('train', 'valid') else 5)
                self.assertEqual(float(result[0, 2]), 50.)


if __name__ == '__main__':
    unittest.main()
