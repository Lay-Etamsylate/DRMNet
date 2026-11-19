import unittest
from unittest.mock import patch

from net.training_setup import configure_training_experiment
from net.model_config import checkpoint_model_config

try:
    import torch
except ImportError:
    torch = None


class DetectorOnlyConfigTests(unittest.TestCase):
    def test_switches_are_saved_and_default_does_not_mutate_input(self):
        original = {'augtype': {'scale': True, 'flip': False}, 'model_variant': 'fusion'}
        cfg = configure_training_experiment(original, 'off', 'off')
        self.assertTrue(original['augtype']['scale'])
        self.assertNotIn('disable_auxiliary_losses', original)
        self.assertTrue(cfg['disable_auxiliary_losses'])
        self.assertFalse(cfg['augtype']['scale'])
        restored = checkpoint_model_config({}, {'model_config': cfg})
        self.assertEqual(configure_training_experiment(restored), restored)
        self.assertTrue(restored['disable_auxiliary_losses'])
        self.assertFalse(restored['augtype']['scale'])
        self.assertEqual(configure_training_experiment(original), original)

    def test_explicit_enable_and_invalid_values(self):
        cfg = configure_training_experiment({'disable_auxiliary_losses': True}, 'on', 'on')
        self.assertFalse(cfg['disable_auxiliary_losses'])
        self.assertTrue(cfg['augtype']['scale'])
        with self.assertRaises(ValueError):
            configure_training_experiment({}, 'unknown')


@unittest.skipIf(torch is None, 'PyTorch and project dependencies are required')
class DetectorOnlyLossTests(unittest.TestCase):
    def test_disabled_auxiliary_never_runs_and_detection_gradient_survives(self):
        from net.sanet import SANet
        model = SANet.__new__(SANet)
        torch.nn.Module.__init__(model)
        model.cfg = {'disable_auxiliary_losses': True}
        model.mode = 'train'
        model.use_rcnn = False
        model.mu = torch.nn.Identity()
        model.logvar = torch.nn.Identity()
        x = torch.ones(1, 4, 2, 2, 2, requires_grad=True)
        with patch.object(model.mu, 'forward', side_effect=AssertionError('mu executed')), \
             patch.object(model.logvar, 'forward', side_effect=AssertionError('logvar executed')), \
             patch('torch.svd', side_effect=AssertionError('SVD executed')):
            model._forward_auxiliary(x)
        self.assertIsNone(model.mu_logvar)
        self.assertEqual(float(model.low_rank_loss_spinal), 0.)
        for name in ('rpn_logits_flat', 'rpn_deltas_flat', 'rpn_labels',
                     'rpn_label_weights', 'rpn_targets', 'rpn_target_weights'):
            setattr(model, name, None)
        p = torch.tensor(2., requires_grad=True)
        with patch('net.sanet.rpn_loss', return_value=(p * 2, p.square(), [])), \
             patch('net.sanet.kl_gaussian_loss', side_effect=AssertionError('KL executed')), \
             patch.object(torch.Tensor, 'cuda', lambda tensor: tensor):
            loss, _, _ = model.loss()
        self.assertEqual(float(loss), 8.)
        loss.backward()
        self.assertEqual(float(p.grad), 6.)

    def test_legacy_configuration_still_runs_auxiliary_branch(self):
        from net.sanet import SANet
        model = SANet.__new__(SANet)
        torch.nn.Module.__init__(model)
        model.cfg = {}
        model.mode = 'train'
        model.feature_dim = 4
        model.mu = torch.nn.Conv3d(4, 4, 1)
        model.logvar = torch.nn.Conv3d(4, 4, 1)
        x = torch.randn(1, 4, 2, 2, 2, requires_grad=True)
        with patch('torch.svd', wraps=torch.svd) as svd:
            model._forward_auxiliary(x)
        svd.assert_called_once()
        self.assertEqual(tuple(model.mu_logvar.shape), (1, 3, 4, 2, 2, 2))
        self.assertTrue(torch.isfinite(model.low_rank_loss_spinal))


if __name__ == '__main__':
    unittest.main()
