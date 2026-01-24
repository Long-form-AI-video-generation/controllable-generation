
import torch
import torch.nn as nn

class ControlAdapter(nn.Module):
    """Adapts multi-modal control features to WAN's DiT"""
    
    def __init__(
        self,
        control_dim=256,      
        hidden_dim=1024,      
        dit_dim=2048,         
        num_controls=6,     
    ):
        super().__init__()
        
       
        self.control_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(control_dim, hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(hidden_dim)
            ) for _ in range(num_controls)
        ])
        
      
        self.fusion = nn.Sequential(
            nn.Linear(num_controls * hidden_dim, dit_dim),
            nn.SiLU(),
            nn.LayerNorm(dit_dim)
        )
        
       
        self.scale = nn.Parameter(torch.zeros(1))
    
    def forward(self, control_features: dict) -> torch.Tensor:
        """
        Args:
            control_features: Dict with keys like 'depth_encoded', 'sketch_encoded'
                             Each shape: (B, C=256, T, H, W)
        Returns:
            control_signal: (B, T*H*W, 2048) to add to DiT features
        """
        B = next(iter(control_features.values())).shape[0]
        
     
        projected = []
        for idx, (key, encoder_output) in enumerate(sorted(control_features.items())):
           
            feat = encoder_output.flatten(2).transpose(1, 2)
            
            proj = self.control_projections[idx](feat)
            projected.append(proj)
        
       
        combined = torch.cat(projected, dim=-1)
        
     
        control_signal = self.fusion(combined)
     
        return control_signal * self.scale.tanh()