import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple



class ConvBlock3D(nn.Module):
   
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: Tuple[int, int, int] = (3, 3, 3),
        stride: Tuple[int, int, int] = (1, 1, 1),
        padding: Tuple[int, int, int] = (1, 1, 1),
        groups: int = 1,
        use_activation: bool = True
    ):
        super().__init__()
        self.conv = nn.Conv3d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False
        )
        # GroupNorm is better than BatchNorm for small batch sizes
        num_groups = min(32, out_channels // 4)
        self.norm = nn.GroupNorm(num_groups, out_channels)
        self.activation = nn.SiLU() if use_activation else nn.Identity()
    
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ResBlock3D(nn.Module):
    
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock3D(channels, channels),
            ConvBlock3D(channels, channels, use_activation=False)
        )
        self.activation = nn.SiLU()
    
    def forward(self, x):
        return self.activation(x + self.block(x))


class SpatialDownsample(nn.Module):
    
    def __init__(self, channels: int, factor: int = 2):
        super().__init__()
        self.conv = nn.Conv3d(
            channels, 
            channels,
            kernel_size=(1, factor, factor),  # Only downsample spatially
            stride=(1, factor, factor),
            padding=0
        )
    
    def forward(self, x):
        return self.conv(x)




class DepthEncoder(nn.Module):
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
       
        self.stem = ConvBlock3D(
            1, 64,
            kernel_size=(1, 7, 7),  
            padding=(0, 3, 3)
        )
        
        self.stage1 = nn.Sequential(
            ConvBlock3D(64, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            ResBlock3D(128),
        )
        
        self.stage2 = nn.Sequential(
            SpatialDownsample(128, factor=2),  # H/2, W/2
            ConvBlock3D(128, 256, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            ResBlock3D(256),
        )
        
   
        self.temporal_smooth = ConvBlock3D(
            256, out_channels,
            kernel_size=(3, 1, 1),  
            padding=(1, 0, 0)
        )
        
        self.output_proj = nn.Conv3d(out_channels, out_channels, 1)
      
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, depth: torch.Tensor) -> torch.Tensor:
      
        if depth.dim() == 4:  
            depth = depth.unsqueeze(1)  
      
        if depth.max() > 1.0:
            depth = depth / 255.0
        
        x = self.stem(depth)          # [B, 64, T, H, W]
        x = self.stage1(x)            # [B, 128, T, H, W]
        x = self.stage2(x)            # [B, 256, T, H/2, W/2]
        x = self.temporal_smooth(x)   # [B, 256, T, H/2, W/2]
        x = self.output_proj(x)       # [B, 256, T, H/2, W/2]
        
        return x



class SketchEncoder(nn.Module):
  
    def __init__(self, out_channels: int = 256):
        super().__init__()
      
        self.stem = nn.Sequential(
            ConvBlock3D(1, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
            ConvBlock3D(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1)),
        )
     
        self.scale1 = ConvBlock3D(64, 64, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        self.scale2 = ConvBlock3D(64, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.scale3 = ConvBlock3D(64, 64, kernel_size=(1, 7, 7), padding=(0, 3, 3))
        
      
        self.fusion = ConvBlock3D(64 * 3, 128)
       
        self.downsample = nn.Sequential(
            SpatialDownsample(128, factor=2),
            ConvBlock3D(128, 256),
            ResBlock3D(256),
        )
        
        
        self.temporal_smooth = ConvBlock3D(
            256, out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        
       
        self.output_proj = nn.Conv3d(out_channels, out_channels, 1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, sketch: torch.Tensor) -> torch.Tensor:
       
        if sketch.dim() == 4:
            sketch = sketch.unsqueeze(1)
        
        
        if sketch.max() > 1.0:
            sketch = sketch / 255.0
        
      
        x = self.stem(sketch) 
        
      
        s1 = self.scale1(x)    # Fine edges
        s2 = self.scale2(x)    # Medium structures
        s3 = self.scale3(x)    # Coarse structures
        
      
        x = torch.cat([s1, s2, s3], dim=1)  # [B, 192, T, H, W]
        x = self.fusion(x)                   # [B, 128, T, H, W]
        
       
        x = self.downsample(x)               # [B, 256, T, H/2, W/2]
        x = self.temporal_smooth(x)          # [B, 256, T, H/2, W/2]
        x = self.output_proj(x)
        
        return x



class MotionEncoder(nn.Module):
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
      
        self.stem = ConvBlock3D(
            2, 64,
            kernel_size=(3, 3, 3), 
            padding=(1, 1, 1)
        )
        
     
        self.temporal_block1 = nn.Sequential(
            ConvBlock3D(64, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            ResBlock3D(128),
        )
        
        self.temporal_block2 = nn.Sequential(
            ConvBlock3D(128, 128, kernel_size=(5, 3, 3), padding=(2, 1, 1)),
            ResBlock3D(128),
        )
        
       
        self.downsample = nn.Sequential(
            SpatialDownsample(128, factor=2),
            ConvBlock3D(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            ResBlock3D(256),
        )
        
        
        self.temporal_refine = nn.Sequential(
            ConvBlock3D(256, 256, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            ConvBlock3D(256, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
        )
        
     
        self.output_proj = nn.Conv3d(out_channels, out_channels, 1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, flow: torch.Tensor) -> torch.Tensor:
        
        # Flow is already 2-channel (dx, dy)
        # Optionally normalize flow magnitude
        # flow = flow / (flow.abs().max() + 1e-6)  # Normalize if needed
        
        x = self.stem(flow)             # [B, 64, T, H, W]
        x = self.temporal_block1(x)     # [B, 128, T, H, W]
        x = self.temporal_block2(x)     # [B, 128, T, H, W]
        x = self.downsample(x)          # [B, 256, T, H/2, W/2]
        x = self.temporal_refine(x)     # [B, 256, T, H/2, W/2]
        x = self.output_proj(x)
        
        return x


class StyleEncoder(nn.Module):
  
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
       
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(16, 128),
            nn.SiLU(),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.GroupNorm(32, 256),
            nn.SiLU(),
        )
       
        self.global_style = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 1, 1]
            nn.Conv2d(256, 256, 1),
        )
        
       
        self.fusion = nn.Conv2d(256 + 256, out_channels, 1)
      
        self.temporal_broadcast = ConvBlock3D(
            out_channels, out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        
      
        self.output_proj = nn.Conv3d(out_channels, out_channels, 1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, reference: torch.Tensor, num_frames: int = 16) -> torch.Tensor:
        
        is_video = reference.dim() == 5
        
        if is_video:
          
            B, T, C, H, W = reference.shape
          
            ref = reference.view(B * T, C, H, W)
        else:
          
            B, C, H, W = reference.shape
            T = 1
            ref = reference
        
      
        local_feat = self.spatial_encoder(ref)  # [B(*T), 256, H/8, W/8]
        global_feat = self.global_style(local_feat)  # [B(*T), 256, 1, 1]
        
        # Broadcast global features to match local spatial size
        global_feat = global_feat.expand_as(local_feat)
        
        # Combine local and global
        combined = torch.cat([local_feat, global_feat], dim=1)  # [B(*T), 512, H/8, W/8]
        features = self.fusion(combined)  # [B(*T), 256, H/8, W/8]
        
        if is_video:
         
            features = features.view(B, T, *features.shape[1:])
        else:
         
            features = features.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        
      
        features = features.permute(0, 2, 1, 3, 4)
        
      
        features = self.temporal_broadcast(features)
        features = self.output_proj(features)
        
        return features



class MaskEncoder(nn.Module):
    
    def __init__(self, out_channels: int = 256):
        super().__init__()
        
      
        self.stem = ConvBlock3D(
            1, 64,
            kernel_size=(1, 7, 7),  
            padding=(0, 3, 3)
        )
        
      
        self.stage1 = ConvBlock3D(64, 128, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        self.stage2 = ConvBlock3D(128, 128, kernel_size=(1, 3, 3), padding=(0, 1, 1))
        
       
        self.downsample = nn.Sequential(
            SpatialDownsample(128, factor=2),
            ConvBlock3D(128, out_channels),
        )
        
       
        self.temporal_prop = ConvBlock3D(
            out_channels, out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        
       
        self.output_proj = nn.Conv3d(out_channels, out_channels, 1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        
        if mask.dim() == 4:
            mask = mask.unsqueeze(1)
        
      
        mask = (mask > 0.5).float()
        
        x = self.stem(mask)         # [B, 64, T, H, W]
        x = self.stage1(x)          # [B, 128, T, H, W]
        x = self.stage2(x)          # [B, 128, T, H, W]
        x = self.downsample(x)      # [B, 256, T, H/2, W/2]
        x = self.temporal_prop(x)   # [B, 256, T, H/2, W/2]
        x = self.output_proj(x)
        
        return x




class PoseEncoder(nn.Module):
   
    def __init__(self, out_channels: int = 256):
        super().__init__()
       
        self.stem = ConvBlock3D(3, 64, kernel_size=(1, 5, 5), padding=(0, 2, 2))
        
        self.stage1 = nn.Sequential(
            ConvBlock3D(64, 128),
            ResBlock3D(128),
        )
        
        self.stage2 = nn.Sequential(
            SpatialDownsample(128, factor=2),
            ConvBlock3D(128, 256),
            ResBlock3D(256),
        )
        
        self.temporal_smooth = ConvBlock3D(
            256, out_channels,
            kernel_size=(3, 1, 1),
            padding=(1, 0, 0)
        )
        
        self.output_proj = nn.Conv3d(out_channels, out_channels, 1)
        nn.init.xavier_uniform_(self.output_proj.weight, gain=0.02)
        nn.init.zeros_(self.output_proj.bias)
    
    def forward(self, pose: torch.Tensor) -> torch.Tensor:
       
        if pose.max() > 1.0:
            pose = pose / 255.0
        
        x = self.stem(pose)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.temporal_smooth(x)
        x = self.output_proj(x)
        
        return x





   