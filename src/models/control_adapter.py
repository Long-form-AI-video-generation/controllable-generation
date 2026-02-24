import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlAdapter(nn.Module):
    

    def __init__(
        self,
        control_dim: int = 256,
        hidden_dim: int = 512,
        dit_dim: int = 2048,
        num_controls: int = 1,
        use_gradient_checkpointing: bool = False,
    ):
        super().__init__()

        self.num_controls = num_controls
        self.use_gradient_checkpointing = use_gradient_checkpointing

       
        self.control_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(control_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1),
            )
            for _ in range(num_controls)
        ])

      
        self.fusion = nn.Sequential(
            nn.Linear(num_controls * hidden_dim, dit_dim),
            nn.SiLU(),
            nn.LayerNorm(dit_dim),
            nn.Dropout(0.1),
        )

        self.temporal_smooth = nn.Conv1d(
            hidden_dim, hidden_dim,
            kernel_size=3, padding=1, groups=hidden_dim
        )

        # self.modality_gates = nn.Parameter(torch.ones(num_controls))

        self.modality_gates = nn.Parameter(torch.randn(num_controls) * 0.1)

        self._init_weights()

        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*70}")
        print("Control Adapter Initialized")
        print(f"{'='*70}")
        print(f"  Input:      {num_controls} modalities × {control_dim} dims")
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

    def forward(self, control_features: dict) -> torch.Tensor:
        """
        Args:
            control_features: dict[str → Tensor(B, 256, T, H, W)]
        Returns:
            control_signal: Tensor(B, T×16×16, dit_dim)
                            — NOT yet zero-conv'd; that happens per-layer in the hook.
        """
        sorted_keys = sorted(control_features.keys())

        if len(sorted_keys) != self.num_controls:
            raise ValueError(
                f"Expected {self.num_controls} controls, got {len(sorted_keys)}: {sorted_keys}"
            )

        first_feat = control_features[sorted_keys[0]]
        # B, C, T, H, W = first_feat.shape
        if first_feat.dim() == 4:
            first_feat = first_feat.unsqueeze(0) 
        B, C, T, H, W = first_feat.shape

        target_spatial = 16  

        projected = []
        for idx, key in enumerate(sorted_keys):
            feat = control_features[key] 

         
            feat = F.adaptive_avg_pool3d(feat, (T, target_spatial, target_spatial))

        
            feat = feat.flatten(2).transpose(1, 2)

            if self.use_gradient_checkpointing and self.training:
                proj = torch.utils.checkpoint.checkpoint(
                    self.control_projections[idx], feat, use_reentrant=False
                )
            else:
                proj = self.control_projections[idx](feat)

          
            gate = torch.sigmoid(self.modality_gates[idx])
            proj = proj.permute(0, 2, 1)               # (B, hidden, T*16*16)
            proj = self.temporal_smooth(proj.float()) # smooth across sequence
            proj = proj.permute(0, 2, 1)              # (B, T*16*16, hidden)
            projected.append(proj * gate)
            # projected.append(proj * gate)

        
        combined = torch.cat(projected, dim=-1) 

        if self.use_gradient_checkpointing and self.training:
            control_signal = torch.utils.checkpoint.checkpoint(
                self.fusion, combined, use_reentrant=False
            )
        else:
            control_signal = self.fusion(combined)

        return control_signal

    def get_modality_weights(self) -> dict:
        gates_for_logging = torch.sigmoid(self.modality_gates).detach().cpu() 
        # modalities = ['sketch']
        modalities = ['sketch'][:self.num_controls] 
        return {mod: float(gates_for_logging[i]) for i, mod in enumerate(modalities)}


if __name__ == '__main__':
    adapter = ControlAdapter()

    B, T, H, W = 1, 8, 64, 64
    dummy_controls = {
        'depth_encoded':  torch.randn(B, 256, T, H, W),
        'mask_encoded':   torch.randn(B, 256, T, H, W),
        'motion_encoded': torch.randn(B, 256, T, H, W),
        'pose_encoded':   torch.randn(B, 256, T, H, W),
        'sketch_encoded': torch.randn(B, 256, T, H, W),
        'style_encoded':  torch.randn(B, 256, T, H, W),
    }

    output = adapter(dummy_controls)
    print(f"Input:  {B} × {T}×{H}×{W} × 256")
    print(f"Output: {output.shape}")  