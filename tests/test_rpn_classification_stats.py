import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required')
class RpnClassificationStatsTests(unittest.TestCase):
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
