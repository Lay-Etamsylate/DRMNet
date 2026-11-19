import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required')
class RpnClassificationStatsTests(unittest.TestCase):
    def test_targets_supervise_each_nodule_with_the_correct_regression(self):
        import numpy as np
        from config import net_config
        from net.layer.rpn_target import make_one_rpn_target
        from net.layer.rpn_nms import rpn_decode
        gt = np.array([[8, 8, 8, 6, 6, 6], [24, 24, 24, 8, 8, 8]], dtype=float)
        windows = np.vstack([gt, [8, 8, 9, 6, 6, 6], [24, 24, 25, 8, 8, 8], [0, 0, 0, 5, 5, 5]])
        image = torch.zeros(1, 32, 32, 32)
        for quota, expected in ((0, 1), (1, 2), (3, 4)):
            cfg = dict(net_config, rpn_positives_per_nodule=quota)
            labels, owners, weights, targets, reg_weights = make_one_rpn_target(
                cfg, 'train', image, windows, gt, np.ones(2))
            pos = labels.numpy().astype(bool)
            self.assertEqual(pos.sum(), expected)
            if quota:
                self.assertEqual(set(owners[pos].tolist()), {0, 1})
            np.testing.assert_array_equal(weights[pos], 1)
            np.testing.assert_array_equal(reg_weights.numpy() > 0, pos)
            decoded = rpn_decode(windows[pos], targets[pos].numpy(), cfg['box_reg_weight'])
            np.testing.assert_allclose(decoded, gt[owners[pos].tolist()], atol=1e-5)

    def test_forced_anchor_collisions_and_unmatched_annotations(self):
        import numpy as np
        from net.layer.rpn_target import assign_per_nodule
        overlap = np.array([[.6, .7, 0, 0], [.5, 0, 0, 0], [0, 0, 0, 1.]])
        indices, owners = assign_per_nodule(overlap, np.array([1, 1, 1, -1]), .8, 3)
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(owners[indices], [1, 0])
        # Reserving the best anchor independently would strand one nodule.
        overlap = np.array([[.9, .8, .8], [.6, 0, 0], [0, .7, .7]])
        indices, owners = assign_per_nodule(overlap, np.ones(3), .5, 1)
        self.assertEqual(set(owners[indices]), {0, 1, 2})

    def test_negative_only_patch_mines_background_and_has_zero_regression(self):
        from net.layer.rpn_loss import rpn_loss
        logits = torch.tensor([[[0.], [1.], [2.]]], requires_grad=True)
        deltas = torch.zeros(1, 3, 6, requires_grad=True)
        labels = torch.zeros(1, 3)
        cls, reg, stats = rpn_loss(logits, deltas, labels, labels + 1, deltas.detach(),
                                   labels, {'num_hard': 2})
        self.assertTrue(torch.isfinite(cls))
        self.assertEqual(stats[1:4], [0, 0, 2])
        self.assertEqual(reg.item(), 0.)
        (cls + reg).backward()
        self.assertEqual(logits.grad[0, 0].item(), 0.)
        self.assertTrue((logits.grad[0, 1:] > 0).all())
        self.assertTrue((deltas.grad == 0).all())

    def test_counts_use_probability_half_without_changing_loss(self):
        from net.layer.rpn_loss import binary_cross_entropy_with_hard_negative_mining
        logits = torch.tensor([[.1], [-.1], [0.], [.2], [-.2], [0.]], requires_grad=True)
        labels = torch.tensor([[1], [1], [1], [0], [0], [0]])
        weights = torch.ones_like(logits)
        loss, pos_ok, pos_n, neg_ok, neg_n = binary_cross_entropy_with_hard_negative_mining(
            logits, labels, weights, batch_size=1, num_hard=0)
        self.assertEqual((int(pos_ok), pos_n, int(neg_ok), neg_n), (2, 3, 1, 3))
        expected = .5 * torch.nn.functional.binary_cross_entropy_with_logits(logits[:3], labels[:3].float())
        expected += .5 * torch.nn.functional.binary_cross_entropy_with_logits(logits[3:], labels[3:].float())
        torch.testing.assert_close(loss, expected)
        gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        torch.testing.assert_close(gradient, expected_gradient)


if __name__ == '__main__':
    unittest.main()
