import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required')
class HeadActivationTests(unittest.TestCase):
    def setUp(self):
        from net.heads import RpnHead, RcnnHead
        self.RpnHead, self.RcnnHead = RpnHead, RcnnHead
        self.cfg = dict(anchors=[[5, 5, 5]], num_class=2, rcnn_crop_size=(1, 1, 1))

    def test_default_preserves_legacy_rcnn_forward_and_state_keys(self):
        head = self.RcnnHead(self.cfg, in_channels=1)
        x = torch.randn(3, 1, 1, 1, 1)
        hidden = torch.relu(head.fc2(torch.relu(head.fc1(x.flatten(1)))))
        logits, deltas = head(x)
        torch.testing.assert_close(logits, head.logit(hidden))
        torch.testing.assert_close(deltas, head.delta(hidden))
        for cls in (self.RcnnHead, self.RpnHead):
            old = cls(self.cfg, in_channels=1)
            alternative = cls(dict(self.cfg, head_activation='leaky_relu'), in_channels=1)
            self.assertEqual(set(old.state_dict()), set(alternative.state_dict()))
            alternative.load_state_dict(old.state_dict(), strict=True)

    def test_negative_rpn_hidden_has_gradient_only_with_leaky(self):
        for activation in ('relu', 'leaky_relu'):
            head = self.RpnHead(dict(self.cfg, head_activation=activation), in_channels=1)
            with torch.no_grad():
                head.conv[0].weight.zero_()
                head.conv[0].bias.fill_(-1)
                head.logits.weight.fill_(.01)
                head.logits.bias.zero_()
            logits, _ = head(torch.ones(1, 1, 2, 2, 2))
            torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones_like(logits)).backward()
            norm = head.conv[0].weight.grad.abs().sum().item()
            self.assertEqual(norm > 0, activation == 'leaky_relu')

    def test_negative_rcnn_hidden_has_gradient_only_with_leaky(self):
        for activation in ('relu', 'leaky_relu'):
            head = self.RcnnHead(dict(self.cfg, head_activation=activation), in_channels=1)
            with torch.no_grad():
                head.fc1.weight.zero_()
                head.fc1.bias.fill_(-1)
                head.fc2.weight.fill_(.01)
                head.fc2.bias.zero_()
                head.logit.weight[0].fill_(.01)
                head.logit.weight[1].fill_(-.01)
                head.logit.bias.zero_()
            logits, _ = head(torch.ones(2, 1, 1, 1, 1))
            torch.nn.functional.cross_entropy(logits, torch.zeros(2, dtype=torch.long)).backward()
            norm = head.fc1.weight.grad.abs().sum().item()
            self.assertEqual(norm > 0, activation == 'leaky_relu')

    def test_invalid_activation_rejected(self):
        with self.assertRaises(ValueError):
            self.RpnHead(dict(self.cfg, head_activation='unknown'), in_channels=1)


if __name__ == '__main__':
    unittest.main()
