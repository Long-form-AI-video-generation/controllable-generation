import unittest

try:
    import torch
except ModuleNotFoundError:  # Local documentation-only environments may omit torch.
    torch = None

if torch is not None:
    from src.mask_models.condition_encoder import (
        MaskConditionEncoder,
        TemporalRefinement,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this local environment")
class MaskConditionEncoderTests(unittest.TestCase):
    def test_shape_and_backward(self):
        torch.manual_seed(7)
        encoder = MaskConditionEncoder(out_channels=32, target_spatial=16)
        control = torch.zeros(2, 1, 4, 128, 128, dtype=torch.long)
        control[:, :, :, 24:104, 63:65] = 12
        output = encoder(control)
        self.assertEqual(output.shape, (2, 32, 4, 16, 16))
        output.square().mean().backward()
        gradients = [p.grad for p in encoder.parameters() if p.requires_grad]
        self.assertTrue(any(g is not None and torch.count_nonzero(g) for g in gradients))

    def test_line_changes_spatial_features(self):
        torch.manual_seed(11)
        encoder = MaskConditionEncoder(out_channels=32, target_spatial=16).eval()
        blank = torch.zeros(1, 1, 2, 128, 128, dtype=torch.long)
        line = blank.clone()
        line[:, :, :, 16:112, 63:66] = 12
        with torch.no_grad():
            difference = (encoder(line) - encoder(blank)).abs()
        self.assertGreater(float(difference.max()), 0.0)
        self.assertGreater(int(torch.count_nonzero(difference)), 0)

    def test_temporal_refinement_has_no_spatial_kernel(self):
        layer = TemporalRefinement(1).eval()
        with torch.no_grad():
            layer.depthwise.weight.fill_(1.0)
            layer.pointwise.weight.fill_(1.0)
            layer.norm.weight.fill_(1.0)
            layer.norm.bias.zero_()
        value = torch.zeros(1, 1, 3, 3, 3)
        value[0, 0, 1, 1, 1] = 1.0
        with torch.no_grad():
            result = layer.depthwise(value)
        self.assertEqual(int(torch.count_nonzero(result[:, :, :, 0, :])), 0)
        self.assertEqual(int(torch.count_nonzero(result[:, :, :, 2, :])), 0)
        self.assertEqual(int(torch.count_nonzero(result[:, :, :, :, 0])), 0)
        self.assertEqual(int(torch.count_nonzero(result[:, :, :, :, 2])), 0)

    def test_invalid_channel_count_is_rejected(self):
        encoder = MaskConditionEncoder(out_channels=32)
        with self.assertRaises(ValueError):
            encoder(torch.zeros(1, 2, 2, 32, 32, dtype=torch.long))


if __name__ == "__main__":
    unittest.main()

