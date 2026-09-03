import io
import unittest

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from src.sketch_models.control_adapter import SketchControlAdapter
    from src.sketch_models.injection import build_padded_control_residual


@unittest.skipIf(torch is None, "PyTorch is not installed in this local environment")
class SketchControlAdapterTests(unittest.TestCase):
    def test_forward_backward_and_state_round_trip(self):
        torch.manual_seed(19)
        adapter = SketchControlAdapter(
            hidden_dim=32,
            dit_dim=48,
            condition_dim=32,
            dropout=0.0,
        )
        control = {"sketch_encoded": torch.zeros(1, 1, 3, 128, 128)}
        control["sketch_encoded"][:, :, :, 32:96, 63:66] = 1.0
        output = adapter(control)
        self.assertEqual(output.shape, (1, 3 * 16 * 16, 48))
        output.square().mean().backward()
        self.assertTrue(any(
            parameter.grad is not None and torch.count_nonzero(parameter.grad)
            for parameter in adapter.parameters()
        ))

        buffer = io.BytesIO()
        torch.save(adapter.state_dict(), buffer)
        buffer.seek(0)
        restored = SketchControlAdapter(
            hidden_dim=32,
            dit_dim=48,
            condition_dim=32,
            dropout=0.0,
        )
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        adapter.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(adapter(control), restored(control))

    def test_wrong_control_key_is_rejected(self):
        adapter = SketchControlAdapter(
            hidden_dim=16,
            dit_dim=24,
            condition_dim=16,
        )
        with self.assertRaises(ValueError):
            adapter({"depth_encoded": torch.zeros(1, 1, 2, 32, 32)})

    def test_gate_is_named_canny(self):
        adapter = SketchControlAdapter(
            hidden_dim=16,
            dit_dim=24,
            condition_dim=16,
        )
        self.assertEqual(set(adapter.get_modality_weights()), {"canny"})

    def test_expected_two_stage_zero_conv_gradient_flow(self):
        adapter = SketchControlAdapter(
            hidden_dim=16,
            dit_dim=24,
            condition_dim=16,
            dropout=0.0,
        )
        projection = torch.nn.Linear(24, 24)
        torch.nn.init.zeros_(projection.weight)
        torch.nn.init.zeros_(projection.bias)
        optimizer = torch.optim.SGD(
            list(adapter.parameters()) + list(projection.parameters()),
            lr=0.1,
        )
        control = {
            "sketch_encoded": torch.zeros(1, 1, 2, 32, 32)
        }
        control["sketch_encoded"][:, :, :, 8:24, 15:17] = 1.0
        hidden = torch.ones(1, 2 * 16 * 16, 24)

        tokens = adapter(control)
        residual = build_padded_control_residual(
            hidden,
            tokens,
            projection,
            source_grid=(2, 16, 16),
            target_grid=(2, 16, 16),
        )
        (hidden + residual).square().mean().backward()
        encoder_gradient = sum(
            float(parameter.grad.norm())
            for parameter in adapter.condition_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertEqual(encoder_gradient, 0.0)
        self.assertGreater(float(projection.weight.grad.norm()), 0.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        tokens = adapter(control)
        residual = build_padded_control_residual(
            hidden,
            tokens,
            projection,
            source_grid=(2, 16, 16),
            target_grid=(2, 16, 16),
        )
        (hidden + residual).square().mean().backward()
        encoder_gradient = sum(
            float(parameter.grad.norm())
            for parameter in adapter.condition_encoder.parameters()
            if parameter.grad is not None
        )
        self.assertGreater(encoder_gradient, 0.0)


if __name__ == "__main__":
    unittest.main()
