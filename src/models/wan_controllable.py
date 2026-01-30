"""
Wrapper for WAN 2.2 with control signal injection
Integrates with the official Wan2.2 codebase
"""

import torch
import torch.nn as nn
from pathlib import Path
import sys
from typing import Optional, Dict


WAN_PATH = Path(__file__).parent.parent / 'Wan2.2'
sys.path.insert(0, str(WAN_PATH))


from wan.models.wan import WanModel
from wan.models.autoencoder import AutoencoderKL
from wan.utils.config import load_config

from models.control_adapter import ControlAdapter


class ControllableWAN(nn.Module):
    """
    WAN 2.2 with controllable generation via control adapter
    """
    
    def __init__(
        self,
        wan_config_path: str,
        wan_checkpoint_path: str,
        vae_checkpoint_path: str,
        control_dim: int = 256,
        num_controls: int = 6,
        use_lora: bool = True,
        lora_rank: int = 8,
        device: str = 'cuda'
    ):
        super().__init__()
        
        self.device = device
        
        print(f"\n{'='*70}")
        print("Initializing Controllable WAN 2.2")
        print(f"{'='*70}")
        
        
        print("  Loading WAN configuration...")
        self.config = load_config(wan_config_path)
        
        
        print("  Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained(vae_checkpoint_path)
        self.vae = self.vae.to(device).eval()
        for param in self.vae.parameters():
            param.requires_grad = False
        
      
        print("  Loading WAN 2.2 model...")
        self.wan = WanModel.from_pretrained(
            wan_checkpoint_path,
            config=self.config
        )
        self.wan = self.wan.to(device)
        
        
        dit_dim = self.config.model.hidden_size  
        
        
        print("  Freezing WAN parameters...")
        for param in self.wan.parameters():
            param.requires_grad = False
        
        print("  Creating control adapter...")
        self.control_adapter = ControlAdapter(
            control_dim=control_dim,
            hidden_dim=1024,
            dit_dim=dit_dim,
            num_controls=num_controls
        ).to(device)
        
        
        if use_lora:
            print(f"  Adding LoRA (rank={lora_rank})...")
            self._add_lora(lora_rank)
       
        print("  Setting up control injection...")
        self._setup_control_hooks()
       
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"{'='*70}")
        print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.2f}M)")
        print(f"  Trainable ratio: {100*trainable_params/total_params:.2f}%")
        print(f"{'='*70}\n")
        
        
        self._control_signal = None
    
    def _add_lora(self, rank: int = 8):
        """Add LoRA adapters to WAN for memory-efficient training"""
        from peft import LoraConfig, get_peft_model
      
        lora_config = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=[
               
                "attn.to_q",
                "attn.to_k", 
                "attn.to_v",
                "attn.to_out.0",
               
                "mlp.fc1",
                "mlp.fc2"
            ],
            lora_dropout=0.1,
            bias="none",
            task_type=None  
        )
        
        
        self.wan = get_peft_model(self.wan, lora_config)
        
     
        self.wan.print_trainable_parameters()
    
    def _setup_control_hooks(self):
       
        if hasattr(self.wan, 'transformer_blocks'):
            blocks = self.wan.transformer_blocks
        elif hasattr(self.wan, 'blocks'):
            blocks = self.wan.blocks
        else:
            raise AttributeError("Cannot find DiT blocks in WAN model")
        
        # Inject control at specific layers (every 4th block)
        self.control_injection_indices = list(range(0, len(blocks), 4))
        
        print(f"  Injecting controls at layers: {self.control_injection_indices}")
        
        for idx in self.control_injection_indices:
            if idx < len(blocks):
                blocks[idx].register_forward_pre_hook(
                    self._control_injection_pre_hook
                )

    def _control_injection_pre_hook(self, module, input):
       
        if self._control_signal is not None:
           
            modified_hidden_states = input[0] + self._control_signal
            
           
            if len(input) > 1:
                return (modified_hidden_states,) + input[1:]
            else:
                return (modified_hidden_states,)
        
        return input
    
    @torch.no_grad()
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        
        video = video * 2.0 - 1.0
        
        latent = self.vae.encode(video).latent_dist.sample()
      
        latent = latent * self.vae.config.scaling_factor
        
        return latent
    
    @torch.no_grad()
    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
       
        latent = latent / self.vae.config.scaling_factor
        
        
        video = self.vae.decode(latent).sample
        
        video = (video + 1.0) / 2.0
        
        return video
    
    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        control_features: Optional[Dict[str, torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        
        if control_features is not None:
            self._control_signal = self.control_adapter(control_features)
        else:
            self._control_signal = None
       
        noise_pred = self.wan(
            latent,
            timesteps,
            encoder_hidden_states,
            **kwargs
        )
        
        
        self._control_signal = None
        
        return noise_pred
    
    def get_trainable_parameters(self):
        """Get all trainable parameters (adapter + LoRA)"""
        return [p for p in self.parameters() if p.requires_grad]


def load_controllable_wan(
    wan_config: str = "Wan2.2/configs/wan2.2_config.yaml",
    wan_checkpoint: str = "checkpoints/wan2.2",
    vae_checkpoint: str = "checkpoints/vae",
    use_lora: bool = True,
    lora_rank: int = 8,
    device: str = 'cuda'
) -> ControllableWAN:
    
    model = ControllableWAN(
        wan_config_path=wan_config,
        wan_checkpoint_path=wan_checkpoint,
        vae_checkpoint_path=vae_checkpoint,
        control_dim=256,
        num_controls=6,
        use_lora=use_lora,
        lora_rank=lora_rank,
        device=device
    )
    
    return model


if __name__ == '__main__':
    
    print("Testing Controllable WAN loading...")
    
    model = load_controllable_wan(
        wan_config="Wan2.2/configs/wan2.2_config.yaml",
        wan_checkpoint="checkpoints/wan2.2",
        vae_checkpoint="checkpoints/vae",
        use_lora=True,
        lora_rank=8
    )
    
    print("\n✓ Model loaded successfully!")
    
  
    B, C, T, H, W = 1, 4, 8, 64, 64
    
    latent = torch.randn(B, C, T, H, W).cuda()
    timesteps = torch.randint(0, 1000, (B,)).cuda()
    encoder_hidden_states = torch.randn(B, 77, 768).cuda()
    
    
    controls = {
        'depth_encoded': torch.randn(B, 256, T, H, W).cuda(),
        'mask_encoded': torch.randn(B, 256, T, H, W).cuda(),
        'motion_encoded': torch.randn(B, 256, T, H, W).cuda(),
        'pose_encoded': torch.randn(B, 256, T, H, W).cuda(),
        'sketch_encoded': torch.randn(B, 256, T, H, W).cuda(),
        'style_encoded': torch.randn(B, 256, T, H, W).cuda(),
    }
    
  
    with torch.no_grad():
        output = model(latent, timesteps, encoder_hidden_states, controls)
    
    print(f"\nForward pass test:")
    print(f"  Input latent: {latent.shape}")
    print(f"  Output: {output.shape}")
    print(f"  ✓ Forward pass successful!")