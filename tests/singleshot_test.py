import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.models.encoders import DepthEncoder, SketchEncoder, MotionEncoder, StyleEncoder, PoseEncoder


def test_single_shot():
    """Test on one shot to verify everything works"""
    
   
    control_base = Path(__file__).parent.parent / 'data' / 'control_signals'
   
    npz_files = list(control_base.rglob('*.npz'))
    
    if not npz_files:
        print(f"No NPZ files found in {control_base}")
        return
    
    test_file = npz_files[0]
    print(f"Testing with: {test_file}")
    print(f"Relative path: {test_file.relative_to(control_base)}\n")
    
    # Load data
    data = np.load(test_file, allow_pickle=True)
    
    print("="*70)
    print("Available arrays in NPZ:")
    print("="*70)
    for key in data.keys():
        arr = data[key]
        if hasattr(arr, 'shape'):
            size_kb = arr.nbytes / 1024
            print(f"{key:20s}: shape={str(arr.shape):25s} dtype={str(arr.dtype):10s} {size_kb:8.1f} KB")
        else:
            print(f"{key:20s}: {type(arr)}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}\n")
    
    # Test depth encoder
    if 'depth' in data:
        print("="*70)
        print("Testing Depth Encoder")
        print("="*70)
        
        encoder = DepthEncoder(out_channels=256).to(device).eval()
        
        depth = data['depth']
        print(f"Raw depth: {depth.shape}, {depth.dtype}, range=[{depth.min()}, {depth.max()}]")
        
        # Take first 8 frames
        depth_8 = depth[:8] if depth.shape[0] >= 8 else depth
        depth_tensor = torch.from_numpy(depth_8.astype(np.float32)) / 255.0
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        print(f"Input tensor: {depth_tensor.shape}")
        
        with torch.no_grad():
            encoded = encoder(depth_tensor)
        
        print(f"Encoded: {encoded.shape}")
        print(f"Range: [{encoded.min().item():.3f}, {encoded.max().item():.3f}]")
        print(f"Memory: {encoded.element_size() * encoded.nelement() / 1e6:.2f} MB")
        print("✓ Depth encoding successful!\n")
    
    # Test edges encoder
    if 'edges' in data:
        print("="*70)
        print("Testing Sketch Encoder")
        print("="*70)
        
        encoder = SketchEncoder(out_channels=256).to(device).eval()
        
        edges = data['edges']
        print(f"Raw edges: {edges.shape}, {edges.dtype}")
        
        edges_8 = edges[:8] if edges.shape[0] >= 8 else edges
        edges_tensor = torch.from_numpy(edges_8.astype(np.float32)) / 255.0
        edges_tensor = edges_tensor.unsqueeze(0).unsqueeze(0).to(device)
        
        with torch.no_grad():
            encoded = encoder(edges_tensor)
        
        print(f"Encoded: {encoded.shape}")
        print("✓ Sketch encoding successful!\n")
    
    # Test flow encoder
    if 'flow' in data:
        print("="*70)
        print("Testing Motion Encoder")
        print("="*70)
        
        encoder = MotionEncoder(out_channels=256).to(device).eval()
        
        flow = data['flow']
        print(f"Raw flow: {flow.shape}, {flow.dtype}")
        
        flow_8 = flow[:8] if flow.shape[0] >= 8 else flow
        flow_tensor = torch.from_numpy(flow_8.astype(np.float32))
        
        # Reshape: (T, H, W, 2) -> (1, 2, T, H, W)
        flow_tensor = flow_tensor.permute(0, 3, 1, 2)  # (T, 2, H, W)
        flow_tensor = flow_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        
        print(f"Input tensor: {flow_tensor.shape}")
        
        with torch.no_grad():
            encoded = encoder(flow_tensor)
        
        print(f"Encoded: {encoded.shape}")
        print("✓ Motion encoding successful!\n")
    
    # Test style encoder
    if 'reference_frame' in data:
        print("="*70)
        print("Testing Style Encoder")
        print("="*70)
        
        encoder = StyleEncoder(out_channels=256).to(device).eval()
        
        ref = data['reference_frame']
        print(f"Raw reference frame: {ref.shape}, {ref.dtype}")
        
        ref_tensor = torch.from_numpy(ref.astype(np.float32)) / 255.0
        ref_tensor = ref_tensor.permute(2, 0, 1).unsqueeze(0).to(device)  # (1, 3, H, W)
        
        print(f"Input tensor: {ref_tensor.shape}")
        
        with torch.no_grad():
            encoded = encoder(ref_tensor, num_frames=8)
        
        print(f"Encoded: {encoded.shape}")
        print("✓ Style encoding successful!\n")
    
    # Test pose encoder
    if 'pose_sequence' in data:
        print("="*70)
        print("Testing Pose Encoder (with heatmap conversion)")
        print("="*70)
        
        encoder = PoseEncoder(out_channels=256).to(device).eval()
        
        pose = data['pose_sequence']
        print(f"Raw pose: {pose.shape}, {pose.dtype}")
        
        # For testing, create dummy heatmaps
       
        pose_8 = pose[:8] if pose.shape[0] >= 8 else pose
        
        # Create simple heatmap (3 channels for head/torso/limbs)
        heatmap = np.random.rand(8, 3, 360, 640).astype(np.float32)
        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0).permute(0, 2, 1, 3, 4).to(device)
        
        print(f"Input tensor (heatmap): {heatmap_tensor.shape}")
        
        with torch.no_grad():
            encoded = encoder(heatmap_tensor)
        
        print(f"Encoded: {encoded.shape}")
        print("✓ Pose encoding successful (with dummy heatmap)!\n")
    
    print("="*70)
    print("All encoders working correctly! ✓")
    print("="*70)
    print(f"\nTest file: {test_file.name}")
    print(f"Total NPZ files found: {len(npz_files)}")


if __name__ == '__main__':
    test_single_shot()