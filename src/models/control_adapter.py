import torch
import torch.nn as nn
import torch.nn.functional as F


# Canonical modality order — the SINGLE source of truth for the positional layout of the
# per-shot `valid_modalities` array. Anything that writes or decodes that array by index
# (encode_controls.py, dataset.py, tests/test-allcontrols.py) MUST import this instead of
# re-listing the names; a divergent copy would silently mask the wrong modality. Spatial
# modalities (everything but style) go through the zero-conv fusion path in alphabetical
# order; style is last and handled by the separate cross-attention pathway.
MODALITY_ORDER = ['depth', 'mask', 'motion', 'pose', 'sketch', 'style']
SPATIAL_MODALITIES = [m for m in MODALITY_ORDER if m != 'style']


class ControlAdapter(nn.Module):

    # Class-level aliases so `ControlAdapter.MODALITY_ORDER` works for callers that have
    # an adapter instance/class but not the module constant.
    MODALITY_ORDER = MODALITY_ORDER
    SPATIAL_MODALITIES = SPATIAL_MODALITIES

    def __init__(
        self,
        control_dim: int = 256,
        hidden_dim: int = 1024,
        dit_dim: int = 2048,
        num_controls: int = 6,
        use_gradient_checkpointing: bool = False,
        style_dim: int = 768,
        text_dim: int = 4096,
        num_style_tokens: int = 4,
    ):
        super().__init__()

        self.num_controls = num_controls
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # BUG 3: style is removed from the spatial fusion path; only the 5 spatial
        # modalities below go through control_projections / fusion / gates.
        self.num_spatial = len(self.SPATIAL_MODALITIES)
        self.style_dim = style_dim
        self.text_dim = text_dim
        self.num_style_tokens = num_style_tokens

        self.control_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(control_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            )
            for _ in range(self.num_spatial)
        ])


        self.fusion = nn.Sequential(
            nn.Linear(self.num_spatial * hidden_dim, dit_dim),
            nn.SiLU(),
            nn.LayerNorm(dit_dim),
            nn.Dropout(0.1),
        )

        # One gate per spatial modality (style no longer has a gate).
        self.modality_gates = nn.Parameter(torch.randn(self.num_spatial) * 0.1)

        # BUG 3: style token pathway. CLIP style embedding -> N text-context tokens,
        # ending in a zero-initialised projection so style influence starts at exactly 0
        # (ControlNet zero-conv style init); injected via cross-attention in wan_controllable.
        self.style_proj = nn.Linear(style_dim, num_style_tokens * text_dim)
        self.style_zero = nn.Linear(text_dim, text_dim)

        self._init_weights()

        # Zero-init the final style projection AFTER _init_weights so it stays exactly zero.
        nn.init.zeros_(self.style_zero.weight)
        nn.init.zeros_(self.style_zero.bias)

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*70}")
        print("Control Adapter Initialized")
        print(f"{'='*70}")
        print(f"  Spatial:    {self.num_spatial} modalities × {control_dim} dims  ({', '.join(self.SPATIAL_MODALITIES)})")
        print(f"  Style:      {style_dim}-dim CLIP embedding -> {num_style_tokens} cross-attn tokens x {text_dim} dims")
        print(f"  Hidden:     {hidden_dim} dims")
        print(f"  Output:     {dit_dim} dims  (zero-conv applied per injection layer)")
        print(f"  Parameters: {total_params:,} ({total_params / 1e6:.2f}M)")
        print(f"{'='*70}\n")

    def _init_weights(self):
        """Xavier uniform for stable early training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, control_features: dict, valid_modalities: dict = None):
        """
        Args:
            control_features: dict[str → Tensor]. Spatial modalities are (B, 256, T, H, W);
                              'style_encoded' is a (B, style_dim) CLIP embedding.
            valid_modalities: optional dict[str → Tensor(B,)] of 1.0/0.0 flags marking which
                              modalities are genuinely present (BUG 1). Absent modalities
                              (e.g. zero-filled motion/pose) are masked to exactly zero BEFORE
                              gating, so neither the fusion nor the gate receives gradient from
                              them.
        Returns:
            (control_signal, style_tokens)
              control_signal: Tensor(B, T×16×16, dit_dim) — NOT yet zero-conv'd (applied per
                              injection layer in the hook).
              style_tokens:   Tensor(B, num_style_tokens, text_dim) — appended to the text
                              cross-attention context; exactly zero at init.
        """
        # BUG 3: style is handled by a separate cross-attention pathway, not spatial fusion.
        style_feat = control_features['style_encoded']
        spatial_keys = sorted(k for k in control_features.keys() if k != 'style_encoded')

        if len(spatial_keys) != self.num_spatial:
            raise ValueError(
                f"Expected {self.num_spatial} spatial controls, got {len(spatial_keys)}: {spatial_keys}"
            )

        first_feat = control_features[spatial_keys[0]]
        B, C, T, H, W = first_feat.shape
        target_spatial = 16

        projected = []
        for idx, key in enumerate(spatial_keys):
            feat = control_features[key]


            feat = F.adaptive_avg_pool3d(feat, (T, target_spatial, target_spatial))


            feat = feat.flatten(2).transpose(1, 2)

            if self.use_gradient_checkpointing and self.training:
                proj = torch.utils.checkpoint.checkpoint(
                    self.control_projections[idx], feat, use_reentrant=False
                )
            else:
                proj = self.control_projections[idx](feat)

            # BUG 1: mask absent modalities to exactly zero BEFORE gating.
            if valid_modalities is not None and key in valid_modalities:
                valid = valid_modalities[key].to(proj.dtype).view(-1, 1, 1)
                proj = proj * valid

            gate = torch.sigmoid(self.modality_gates[idx])
            projected.append(proj * gate)


        combined = torch.cat(projected, dim=-1)

        if self.use_gradient_checkpointing and self.training:
            control_signal = torch.utils.checkpoint.checkpoint(
                self.fusion, combined, use_reentrant=False
            )
        else:
            control_signal = self.fusion(combined)

        # BUG 3: style token pathway (zero at init via self.style_zero).
        if style_feat.dim() > 2:
            style_feat = style_feat.reshape(style_feat.shape[0], -1)
        style_tokens = self.style_proj(style_feat)
        style_tokens = style_tokens.view(-1, self.num_style_tokens, self.text_dim)
        style_tokens = self.style_zero(style_tokens)

        # BUG 1: mask absent style to exactly zero as well.
        if valid_modalities is not None and 'style_encoded' in valid_modalities:
            valid = valid_modalities['style_encoded'].to(style_tokens.dtype).view(-1, 1, 1)
            style_tokens = style_tokens * valid

        return control_signal, style_tokens

    def get_modality_weights(self) -> dict:
        """Inspect learned modality importance (for logging/debugging)."""
        gates = torch.sigmoid(self.modality_gates).detach().cpu()
        return {mod: float(gates[i]) for i, mod in enumerate(self.SPATIAL_MODALITIES)}

    def get_modality_logits(self) -> dict:
        """Raw (pre-sigmoid) gate logits, per spatial modality (BUG 2 logging)."""
        logits = self.modality_gates.detach().cpu()
        return {mod: float(logits[i]) for i, mod in enumerate(self.SPATIAL_MODALITIES)}


if __name__ == '__main__':
    adapter = ControlAdapter()

    B, T, H, W = 1, 8, 64, 64
    dummy_controls = {
        'depth_encoded':  torch.randn(B, 256, T, H, W),
        'mask_encoded':   torch.randn(B, 256, T, H, W),
        'motion_encoded': torch.randn(B, 256, T, H, W),
        'pose_encoded':   torch.randn(B, 256, T, H, W),
        'sketch_encoded': torch.randn(B, 256, T, H, W),
        'style_encoded':  torch.randn(B, 768),
    }

    control_signal, style_tokens = adapter(dummy_controls)
    print(f"Input:  {B} × {T}×{H}×{W} × 256")
    print(f"Control signal: {control_signal.shape}")
    print(f"Style tokens:   {style_tokens.shape}")