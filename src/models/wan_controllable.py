
import torch
import torch.nn as nn
from pathlib import Path
import sys

import time

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent  
WAN_PATH = project_root / 'Wan2.2'

if not WAN_PATH.exists():
    raise RuntimeError(f"WAN directory not found at {WAN_PATH}")


sys.path.insert(0, str(WAN_PATH))


from wan.modules import WanModel, Wan2_2_VAE, T5EncoderModel
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.configs import WAN_CONFIGS


sys.path.insert(0, str(current_file.parent.parent))  
from models.control_adapter import ControlAdapter


class ControllableWAN(nn.Module):
   
    
    def __init__(
        self,
        checkpoint_dir: str,
        device: str = 'cuda',
        control_injection_layers: list = [0, 4, 8, 12, 16, 20, 24, 28],
        spatial_downsample: int = 16,
    ):
        super().__init__()
        
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir).resolve()  
        self.control_injection_layers = control_injection_layers
        
        if not self.checkpoint_dir.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {self.checkpoint_dir}")
        
        print(f"\n{'='*70}")
        print("Loading Controllable WAN 2.2")
        print(f"{'='*70}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
       
        print(f"\n{'='*70}")
        print("Loading Controllable WAN 2.2")
        print(f"{'='*70}")
        
        
        print("  [1/4] Loading VAE...")
        self.vae = self._load_vae()
        
      
        print("  [2/4] Loading WAN DiT...")
        self.wan = self._load_wan()
        
      
        print("  [3/4] Loading T5 text encoder...")
        self.text_encoder, self.tokenizer = self._load_text_encoder()
        
       
        print("  [4/4] Creating control adapter...")
        self.control_adapter = ControlAdapter(
            control_dim=256,
            hidden_dim=1024,
            dit_dim=self.wan.config.dim,  
            num_controls=6,
            use_gradient_checkpointing= True
            # spatial_downsample_size=spatial_downsample
        ).to(device)
        
      
        self._setup_control_hooks()
        
        
        self._control_signal = None
        
      
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = total - trainable
        
        print(f"{'='*70}")
        print(f"  Total params: {total:,} ({total/1e9:.2f}B)")
        print(f"  Frozen (WAN+VAE+T5): {frozen:,} ({frozen/1e9:.2f}B)")
        print(f"  Trainable (Adapter): {trainable:,} ({trainable/1e6:.1f}M)")
        print(f"  Control injection at layers: {control_injection_layers}")
        print(f"{'='*70}\n")
    
    def _load_vae(self):
        """Load Wan2.2 VAE"""
        from wan.modules.vae2_2 import Wan2_2_VAE
        
        vae_path = self.checkpoint_dir / 'Wan2.2_VAE.pth'
        
        if not vae_path.exists():
            raise FileNotFoundError(f"VAE not found at {vae_path}")
        
        print(f"     Loading VAE from {vae_path}...")
        
        device_vae= torch.device('cuda:1')
        # device_cpu = 'cpu'
        vae = Wan2_2_VAE(
            vae_pth=str(vae_path),  
            z_dim=48,
            c_dim=160,
            dim_mult=[1, 2, 4, 4],
            temperal_downsample=[False, True, True],
            dtype=torch.float16,  
            device=device_vae
        )
        
       
        for param in vae.model.parameters():
            param.requires_grad = False
        
        print(f"   VAE loaded successfully")
        return vae

    def _load_wan(self):
        """Load WAN DiT model - initially on CPU if offload mode"""
        from wan.configs import WAN_CONFIGS
        from safetensors.torch import load_file
        import json
        
        config_path = self.checkpoint_dir / 'config.json'
        with open(config_path) as f:
            model_config = json.load(f)
        
        print(f"     Model config: {model_config.get('num_layers', 32)} layers")
        offload_model= True
        # Create WAN model
        wan = WanModel(
            model_type=model_config.get('model_type', 'ti2v'),
            patch_size=tuple(model_config.get('patch_size', [1, 2, 2])),
            text_len=model_config.get('text_len', 512),
            in_dim=model_config.get('in_dim', 16),
            dim=model_config.get('dim', 2048),
            ffn_dim=model_config.get('ffn_dim', 8192),
            freq_dim=model_config.get('freq_dim', 256),
            text_dim=model_config.get('text_dim', 4096),
            out_dim=model_config.get('out_dim', 16),
            num_heads=model_config.get('num_heads', 16),
            num_layers=model_config.get('num_layers', 32),
            window_size=tuple(model_config.get('window_size', [-1, -1])),
            qk_norm=model_config.get('qk_norm', True),
            cross_attn_norm=model_config.get('cross_attn_norm', True),
            eps=model_config.get('eps', 1e-6)
        )
        
        print("     Loading WAN weights from safetensors...")
        
        index_path = self.checkpoint_dir / 'diffusion_pytorch_model.safetensors.index.json'
        with open(index_path) as f:
            index = json.load(f)
        
        state_dict = {}
        weight_map = index['weight_map']
        shard_files = set(weight_map.values())
        
        for shard_file in sorted(shard_files):
            shard_path = self.checkpoint_dir / shard_file
            print(f"     Loading {shard_file}...")
            shard_state = load_file(str(shard_path))
            state_dict.update(shard_state)
        
        wan.load_state_dict(state_dict)
        
        # If not offloading, move to device now
        if not offload_model:
            wan = wan.to(self.device)
        
        wan = wan.eval()
        
        # Freeze all parameters
        for param in wan.parameters():
            param.requires_grad = False
        
        print(f"   WAN loaded ({model_config.get('num_layers', 32)} layers)")
        return wan
    
    def _load_text_encoder(self):
        """Load T5 text encoder"""
        from wan.modules.t5 import T5EncoderModel
        
        t5_path = self.checkpoint_dir / 'models_t5_umt5-xxl-enc-bf16.pth'
        
        if not t5_path.exists():
            raise FileNotFoundError(f"T5 weights not found at {t5_path}")
        
        print(f"     Loading T5 from {t5_path}...")
        
     
        device_cpu = 'cpu'
        text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=device_cpu,
            checkpoint_path=str(t5_path),
            tokenizer_path='google/umt5-xxl'
        )
        
        tokenizer = text_encoder.tokenizer
        
       
        self.t5_device = device_cpu
        
        print(f"  T5 encoder loaded on {device_cpu} (to save GPU memory)")
        return text_encoder, tokenizer
    
    def _setup_control_hooks(self):
        """Setup forward hooks for control injection"""
        self.hooks = []
        
        for layer_idx in self.control_injection_layers:
            if layer_idx < len(self.wan.blocks):
                hook = self.wan.blocks[layer_idx].register_forward_pre_hook(
                    self._control_injection_hook
                )
                self.hooks.append(hook)
    
    def _control_injection_hook(self, module, input):
        """Hook to inject control signal before WanAttentionBlock"""
        if self._control_signal is not None:
            x = input[0]  # [B, L, C]
            
            
            B, L, C = x.shape
            B_c, N_c, C_c = self._control_signal.shape
            
            if N_c == L:
                
                x = x + self._control_signal
            else:
              
                control = self._control_signal.permute(0, 2, 1)  # [B, C, N_c]
                control = torch.nn.functional.interpolate(
                    control, 
                    size=L, 
                    mode='linear', 
                    align_corners=False
                )
                control = control.permute(0, 2, 1)  # [B, L, C]
                x = x + control
            
            
            return (x,) + input[1:]
        
        return input
    
    def encode_video(self, video: torch.Tensor) -> torch.Tensor:
        
        
        vae_device = torch.device("cuda:1")
        output_device = torch.device("cuda:0")
        
    
        video = video.to(vae_device)
        
        self.vae.device = vae_device
        
        with torch.no_grad():
            video_list = [video[i] for i in range(video.shape[0])]
            latents = self.vae.encode(video_list)
            latent = torch.stack(latents)
        
        
        latent = latent.to(output_device)
        
        # self.vae.device = 'cpu'
        # torch.cuda.empty_cache()
        
        return latent

    def decode_video(self, latent: torch.Tensor) -> torch.Tensor:
        
        vae_device = torch.device("cuda:1")
        output_device = torch.device("cuda:0")
        
    
        latent = latent.to(vae_device)
        
        self.vae.device = vae_device
        with torch.no_grad():
            
            latent_list = [latent[i] for i in range(latent.shape[0])]
            
        
            videos = self.vae.decode(latent_list)
            
          
            video = torch.stack(videos)

        video = video.to(output_device)
        
        # self.vae.device = 'cpu'
        # torch.cuda.empty_cache()
        
        return video
    def encode_text(self, prompts: list) -> list:
        
        # Determine best device for T5
        if torch.cuda.device_count() > 1:
            t5_compute_device = torch.device('cuda:1') 
        else:
            t5_compute_device = torch.device('cuda:0')  
        
        with torch.no_grad():
            
            self.text_encoder.model.to(t5_compute_device)
           
            text_list = self.text_encoder(prompts, device=t5_compute_device)
        
            self.text_encoder.model.cpu()
            torch.cuda.empty_cache()
        
     
        padded = []
        for t in text_list:
          
            if t.shape[0] < 512:
                padding = torch.zeros(
                    512 - t.shape[0], 
                    t.shape[1], 
                    device=t.device, 
                    dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            
            
            t = t.to(self.device, dtype=torch.float32)
            padded.append(t)
        
        return padded
    
    def forward(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        prompts: list,
        control_features: dict = None,
    ) -> torch.Tensor:
        
        offload_model= True
      
        if offload_model:
            t0 = time.time()
            self.wan.to(self.device)
            torch.cuda.empty_cache()
            print(f"  WAN load: {time.time()-t0:.1f}s")
    
        latent = latent.to(self.device)
        timesteps = timesteps.to(self.device)
       
        if control_features is not None:
            t0 = time.time()
            controls_device = {k: v.to(self.device) for k, v in control_features.items()}
            print(f"  Control move: {time.time()-t0:.1f}s")

            t0 = time.time()
            self._control_signal = self.control_adapter(controls_device)
            print(f"  Control adapter: {time.time()-t0:.1f}s")
        else:
            self._control_signal = None
      
        t0 = time.time()
        text_embeddings = self.encode_text(prompts)
        print(f"  Text encode: {time.time()-t0:.1f}s")

        context = [text_embeddings[i] for i in range(len(prompts))]
    
        x = [latent[i] for i in range(latent.shape[0])]
      
        B, C, T, H, W = latent.shape
        patch_t, patch_h, patch_w = self.wan.patch_size
        t0 = time.time()
        seq_len_actual = (T // patch_t) * (H // patch_h) * (W // patch_w)
    
        
        seq_len = ((seq_len_actual + 63) // 64) * 64
        noise_pred = self.wan(
            x=x,
            t=timesteps,
            context=context,
            seq_len=seq_len,
            y=None
        )
        print(f"  WAN forward: {time.time()-t0:.1f}s")
  
        noise_pred = torch.stack([n for n in noise_pred])
        
        self._control_signal = None
        
        return noise_pred
    
    def get_trainable_parameters(self):
      
        return [p for p in self.control_adapter.parameters() if p.requires_grad]


def test_controllable_wan():
   
    
    print("\n" + "="*70)
    print("Testing Controllable WAN")
    print("="*70)
   
    model = ControllableWAN(
        checkpoint_dir='Wan2.2/Wan2.2-TI2V-5B',
        device='cuda'
    )
    
    
    B = 1
    latent = torch.randn(B, 48, 8, 32, 32).cuda()  
    timesteps = torch.randint(0, 1000, (B,)).cuda()
    prompts = ["A cat playing with a ball"]
    
    # Dummy controls
    controls = {
        'depth_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
        'sketch_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
        'motion_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
        'style_encoded': torch.randn(B, 256, 8, 32, 32).cuda(),
        'pose_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
        'mask_encoded': torch.randn(B, 256, 8, 128, 128).cuda(),
    }
    
   
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(latent, timesteps, prompts, controls)
    
    print(f"\nForward pass successful!")
    print(f"   Input latent: {latent.shape}")
    print(f"   Output: {output.shape}")
    print(f"   Trainable params: {sum(p.numel() for p in model.get_trainable_parameters()):,}")


if __name__ == '__main__':
    test_controllable_wan()