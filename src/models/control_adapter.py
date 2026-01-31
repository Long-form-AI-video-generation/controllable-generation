# models/control_adapter.py
import torch
import torch.nn as nn
import torch.nn.functional as F 
class ControlAdapter(nn.Module):
    """
    Adapts multi-modal control features to WAN's DiT dimension
  
    """
    
    def __init__(
        self,
        control_dim: int = 256,      
        hidden_dim: int = 1024,      
        dit_dim: int = 2048,        
        num_controls: int = 6,      
        use_gradient_checkpointing: bool = False
    ):
        super().__init__()
        
        self.num_controls = num_controls
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
       
        self.control_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(control_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ) for _ in range(num_controls)
        ])
        
      
        self.fusion = nn.Sequential(
            nn.Linear(num_controls * hidden_dim, dit_dim),
            nn.SiLU(),
            nn.LayerNorm(dit_dim),
            nn.Dropout(0.1)
        )
        
       
        self.scale = nn.Parameter(torch.zeros(1))
        
        
        self.modality_gates = nn.Parameter(torch.ones(num_controls))
        
       
        self._init_weights()
        
        total_params = sum(p.numel() for p in self.parameters())
        print(f"\n{'='*70}")
        print("Control Adapter Initialized")
        print(f"{'='*70}")
        print(f"  Input: {num_controls} modalities × {control_dim} dims")
        print(f"  Hidden: {hidden_dim} dims")
        print(f"  Output: {dit_dim} dims")
        print(f"  Parameters: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"{'='*70}\n")
    
    def _init_weights(self):
        """Initialize weights with small values for stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, control_features: dict) -> torch.Tensor:
        """
        Args:
            control_features: Dict with encoded controls
                            Each shape: (B, 256, T, H, W)
        Returns:
            control_signal: (B, N, 2048) where N=T*H'*W' (spatially downsampled)
        """
        # Sort keys for consistent ordering
        sorted_keys = sorted(control_features.keys())
        
        if len(sorted_keys) != self.num_controls:
            raise ValueError(
                f"Expected {self.num_controls} controls, got {len(sorted_keys)}"
            )
        
        first_feat = control_features[sorted_keys[0]]
        B, C, T, H, W = first_feat.shape
        
      
        target_spatial = 16  
        
        projected = []
        
        for idx, key in enumerate(sorted_keys):
            feat = control_features[key]  
            # Spatial pooling: [B, 256, T, H, W] → [B, 256, T, 16, 16]
            feat = F.adaptive_avg_pool3d(feat, (T, target_spatial, target_spatial))
            
           
            feat = feat.flatten(2).transpose(1, 2) 
            if self.use_gradient_checkpointing and self.training:
                proj = torch.utils.checkpoint.checkpoint(
                    self.control_projections[idx],
                    feat,
                    use_reentrant=False
                )
            else:
                proj = self.control_projections[idx](feat)
         
            gate = torch.sigmoid(self.modality_gates[idx])
            proj = proj * gate
            
            projected.append(proj)
        
        
        combined = torch.cat(projected, dim=-1)  # [B, N, 6×1024]
        
      
        if self.use_gradient_checkpointing and self.training:
            control_signal = torch.utils.checkpoint.checkpoint(
                self.fusion,
                combined,
                use_reentrant=False
            )
        else:
            control_signal = self.fusion(combined)
        
        # Scale
        control_signal = control_signal * self.scale.tanh()
        
        return control_signal  # [B, T×16×16, 2048] = [B, 2048, 2048]
    
    def get_modality_weights(self) -> dict:
        """Get learned importance of each modality"""
        gates = torch.sigmoid(self.modality_gates).detach().cpu()
        modalities = ['depth', 'mask', 'motion', 'pose', 'sketch', 'style']
        return {mod: float(gates[i]) for i, mod in enumerate(modalities)}


if __name__ == '__main__':
    
    adapter = ControlAdapter()
    
    # Create dummy inputs for test
    B, T, H, W = 1, 8, 64, 64
    dummy_controls = {
        'depth_encoded': torch.randn(B, 256, T, H, W),
        'mask_encoded': torch.randn(B, 256, T, H, W),
        'motion_encoded': torch.randn(B, 256, T, H, W),
        'pose_encoded': torch.randn(B, 256, T, H, W),
        'sketch_encoded': torch.randn(B, 256, T, H, W),
        'style_encoded': torch.randn(B, 256, T, H, W),
    }
    
   
    output = adapter(dummy_controls)
    print(f"Input: {B} × {T}×{H}×{W} patches × 256 dims")
    print(f"Output: {output.shape}")  # Should be (1, 32768, 2048)
    