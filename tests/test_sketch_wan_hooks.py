import unittest

try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    torch = None
    nn = None

if torch is not None:
    from src.sketch_models.injection import (
        build_padded_control_residual,
        normalize_control_residual,
        resize_control_tokens,
        validate_cfg_control_policy,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in this local environment")
class SketchWanHookTests(unittest.TestCase):
    def test_training_grid_and_padding(self):
        control = torch.ones(1, 8 * 16 * 16, 3)
        hidden = torch.ones(1, 64, 3)
        projection = nn.Linear(3, 3)
        with torch.no_grad():
            projection.weight.zero_()
            projection.bias.fill_(2.0)
        result = build_padded_control_residual(
            hidden,
            control,
            projection,
            source_grid=(8, 16, 16),
            target_grid=(2, 4, 4),
        )
        self.assertEqual(result.shape, hidden.shape)
        self.assertGreater(int(torch.count_nonzero(result[:, :32])), 0)
        self.assertEqual(int(torch.count_nonzero(result[:, 32:])), 0)

    def test_rectangular_inference_grid_and_padding(self):
        control = torch.ones(1, 81 * 16 * 16, 2)
        hidden = torch.ones(1, 32768, 2)
        result = build_padded_control_residual(
            hidden,
            control,
            nn.Identity(),
            source_grid=(81, 16, 16),
            target_grid=(21, 30, 52),
        )
        self.assertEqual(int(torch.count_nonzero(result[:, 32760:])), 0)

    def test_strength_scales_monotonically(self):
        hidden = torch.ones(1, 4, 2)
        residual = torch.ones_like(hidden)
        magnitudes = []
        for strength in (0.0, 0.25, 0.5, 1.0):
            value = normalize_control_residual(
                hidden,
                residual,
                control_strength=strength,
            )
            magnitudes.append(float(value.norm()))
        self.assertEqual(magnitudes[0], 0.0)
        self.assertEqual(magnitudes, sorted(magnitudes))

    def test_zero_projection_receives_first_step_gradient(self):
        projection = nn.Linear(3, 3)
        nn.init.zeros_(projection.weight)
        nn.init.zeros_(projection.bias)
        control = torch.ones(1, 8 * 16 * 16, 3, requires_grad=True)
        hidden = torch.ones(1, 64, 3)
        residual = build_padded_control_residual(
            hidden,
            control,
            projection,
            source_grid=(8, 16, 16),
            target_grid=(2, 4, 4),
        )
        (hidden + residual).square().mean().backward()
        self.assertGreater(float(projection.weight.grad.norm()), 0.0)
        self.assertEqual(float(control.grad.norm()), 0.0)

    def test_resize_validates_source_grid(self):
        with self.assertRaises(ValueError):
            resize_control_tokens(
                torch.zeros(1, 15, 2),
                source_grid=(1, 4, 4),
                target_grid=(1, 2, 2),
            )

    def test_cfg_policy_is_explicit(self):
        self.assertEqual(
            validate_cfg_control_policy("both_text_branches"),
            "both_text_branches",
        )
        with self.assertRaises(ValueError):
            validate_cfg_control_policy("conditional_only")


if __name__ == "__main__":
    unittest.main()
