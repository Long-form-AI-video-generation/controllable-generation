
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import sys
import time
from typing import Dict, List, Optional

sys.path.append(str(Path(__file__).parent.parent))
from models.encoders import (
    DepthEncoder, SketchEncoder, MotionEncoder, 
    StyleEncoder, PoseEncoder, MaskEncoder
)


class ControlEncoderProcessor:
    """Process all control NPZ files through encoders"""
    
    def __init__(
        self,
        control_base_dir: str,
        output_dir: str,
        device: str = 'cuda',
        num_frames: int = 8,
        resolution: tuple = (256, 256)
    ):
        self.control_base = Path(control_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = device
        self.num_frames = num_frames
        self.resolution = resolution
        
      
        self.encoders = self._init_encoders()
        
        
        self.npz_files = self._find_all_npz_files()
        print(f"Found {len(self.npz_files)} NPZ files to process\n")
    
    def _init_encoders(self) -> Dict[str, nn.Module]:
        """Initialize all control encoders"""
        print("="*70)
        print("Initializing Encoders")
        print("="*70)
        
        encoders = {
            'depth': DepthEncoder(out_channels=256),
            'sketch': SketchEncoder(out_channels=256),
            'motion': MotionEncoder(out_channels=256),
            'style': StyleEncoder(out_channels=256),
            'pose': PoseEncoder(out_channels=256),
            'mask': MaskEncoder(out_channels=256)
        }
        
        for name, encoder in encoders.items():
            encoder = encoder.to(self.device).eval()
            encoders[name] = encoder
            param_count = sum(p.numel() for p in encoder.parameters()) / 1e6
            print(f"  ✓ {name:10s}: {param_count:6.2f}M params")
        
        print("="*70 + "\n")
        return encoders
    
    def _find_all_npz_files(self) -> list:
        """Find all NPZ files recursively"""
        files = []
        
        for npz_path in self.control_base.rglob('*.npz'):
            rel_path = npz_path.relative_to(self.control_base)
            output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_encoded.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            files.append({
                'input_path': npz_path,
                'output_path': output_path,
                'video_id': rel_path.parent.name,
                'shot_name': rel_path.stem
            })
        
        return files
    
    def _sample_frames(self, data: np.ndarray, target_frames: int) -> np.ndarray:
        """Sample frames uniformly"""
        current_frames = data.shape[0]
        if current_frames <= target_frames:
         
            if current_frames < target_frames:
                pad_width = [(0, target_frames - current_frames)] + [(0, 0)] * (data.ndim - 1)
                data = np.pad(data, pad_width, mode='edge')
            return data
        
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return data[indices]
    
    def _prepare_depth(self, depth: np.ndarray) -> torch.Tensor:
        """Prepare depth: (T, H, W) -> (1, 1, T, H, W)"""
        depth = self._sample_frames(depth, self.num_frames)
        
        resized = []
        for t in range(depth.shape[0]):
            frame = cv2.resize(depth[t], self.resolution, interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        depth = np.stack(resized)
        
        tensor = torch.from_numpy(depth.astype(np.float32)) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def _prepare_edges(self, edges: np.ndarray) -> torch.Tensor:
        """Prepare edges: (T, H, W) -> (1, 1, T, H, W)"""
        edges = self._sample_frames(edges, self.num_frames)
        
        resized = []
        for t in range(edges.shape[0]):
            frame = cv2.resize(edges[t], self.resolution, interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        edges = np.stack(resized)
        
        tensor = torch.from_numpy(edges.astype(np.float32)) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def _prepare_flow(self, flow: np.ndarray) -> torch.Tensor:
        """Prepare flow: (T, H, W, 2) -> (1, 2, T, H, W) with safety checks"""
        
       
        if flow.shape[0] == 0:
          
            return torch.zeros(1, 2, self.num_frames, *self.resolution, device=self.device)
       
        if not np.isfinite(flow).all():
            
            flow = np.nan_to_num(flow, nan=0.0, posinf=0.0, neginf=0.0)
        
        
        flow = self._sample_frames(flow, self.num_frames)
        
       
        resized = []
        h_scale = self.resolution[1] / flow.shape[1]
        w_scale = self.resolution[0] / flow.shape[2]
        
        for t in range(flow.shape[0]):
            try:
                flow_x = cv2.resize(
                    flow[t, :, :, 0].astype(np.float32), 
                    self.resolution, 
                    interpolation=cv2.INTER_LINEAR
                )
                flow_y = cv2.resize(
                    flow[t, :, :, 1].astype(np.float32), 
                    self.resolution, 
                    interpolation=cv2.INTER_LINEAR
                )
                flow_x *= w_scale
                flow_y *= h_scale
                resized.append(np.stack([flow_x, flow_y], axis=-1))
            except Exception as e:
                
                resized.append(np.zeros((self.resolution[1], self.resolution[0], 2), dtype=np.float32))
        
        flow = np.stack(resized)
        tensor = torch.from_numpy(flow.astype(np.float32))
        tensor = tensor.permute(0, 3, 1, 2).unsqueeze(0).permute(0, 2, 1, 3, 4)
        return tensor.to(self.device)
    
    def _prepare_reference(self, ref: np.ndarray) -> torch.Tensor:
        """Prepare reference: (H, W, 3) -> (1, 3, H, W)"""
        ref = cv2.resize(ref, self.resolution, interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(ref.astype(np.float32)) / 255.0
        tensor = tensor.permute(2, 0, 1).unsqueeze(0)
        return tensor.to(self.device)
    
    def _keypoints_to_heatmap(self, keypoints: np.ndarray, sigma: float = 5.0) -> np.ndarray:
        """Convert pose keypoints to heatmaps with safety checks"""
        T, num_joints, _ = keypoints.shape
        H, W = self.resolution[1], self.resolution[0]
        
      
        heatmaps = np.zeros((T, 3, H, W), dtype=np.float32)
        
        head_joints = [0, 1, 2, 3, 4]
        torso_joints = [5, 6, 11, 12]
        limb_joints = [7, 8, 9, 10, 13, 14, 15, 16]
        groups = [head_joints, torso_joints, limb_joints]
        
        for t in range(T):
            for ch, joint_group in enumerate(groups):
                for joint_idx in joint_group:
                    if joint_idx >= num_joints:
                        continue
                    
                    x, y, conf = keypoints[t, joint_idx]
                    
                  
                    if not np.isfinite(x) or not np.isfinite(y) or not np.isfinite(conf):
                        continue  
                    
                    if conf < 0.3:
                        continue  
                    
                   
                    x = np.clip(x, 0.0, 1.0)
                    y = np.clip(y, 0.0, 1.0)
                    
                   
                    x_norm = int(x * W)
                    y_norm = int(y * H)
                    
                    
                    x_norm = np.clip(x_norm, 0, W - 1)
                    y_norm = np.clip(y_norm, 0, H - 1)
                    
                    
                    y_grid, x_grid = np.ogrid[:H, :W]
                    gaussian = np.exp(-((x_grid - x_norm)**2 + (y_grid - y_norm)**2) / (2 * sigma**2))
                    heatmaps[t, ch] = np.maximum(heatmaps[t, ch], gaussian)
        
        return heatmaps
    
    def _prepare_pose(self, pose: np.ndarray) -> torch.Tensor:
        """Prepare pose: (T, 17, 3) -> (1, 3, T, H, W)"""
        pose = self._sample_frames(pose, self.num_frames)
        heatmaps = self._keypoints_to_heatmap(pose)
        tensor = torch.from_numpy(heatmaps)
        tensor = tensor.unsqueeze(0).permute(0, 2, 1, 3, 4)
        return tensor.to(self.device)

    def _prepare_mask(self, mask: np.ndarray) -> torch.Tensor: 
        """Prepare mask: (T, H, W) -> (1, 1, T, H, W)"""
        mask = self._sample_frames(mask, self.num_frames)
        
        resized = []
        for t in range(mask.shape[0]):
            frame = cv2.resize(mask[t], self.resolution, interpolation=cv2.INTER_NEAREST)
            resized.append(frame)
        mask = np.stack(resized)
        
        tensor = torch.from_numpy(mask.astype(np.float32)) / 255.0
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        return tensor.to(self.device)
    
    def process_single_file(self, file_info: dict) -> dict:
        """Process a single NPZ file"""
        result = {
            'success': False,
            'encodings': 0,
            'size_mb': 0,
            'errors': []
        }
        
        try:
            
            if file_info['output_path'].exists():
                result['success'] = True
                result['size_mb'] = file_info['output_path'].stat().st_size / 1e6
                return result
           
            data = np.load(file_info['input_path'], allow_pickle=True)
            encoded = {}
            
            with torch.no_grad():
               
                if 'depth' in data:
                    try:
                        tensor = self._prepare_depth(data['depth'])
                        enc = self.encoders['depth'](tensor)
                        encoded['depth_encoded'] = enc.cpu().numpy().astype(np.float16)
                    except Exception as e:
                        result['errors'].append(f"depth: {str(e)[:50]}")
                
               
                if 'edges' in data:
                    try:
                        tensor = self._prepare_edges(data['edges'])
                        enc = self.encoders['sketch'](tensor)
                        encoded['sketch_encoded'] = enc.cpu().numpy().astype(np.float16)
                    except Exception as e:
                        result['errors'].append(f"edges: {str(e)[:50]}")
               
                if 'flow' in data and data['flow'].shape[0] > 0:
                    try:
                        tensor = self._prepare_flow(data['flow'])
                        enc = self.encoders['motion'](tensor)
                        encoded['motion_encoded'] = enc.cpu().numpy().astype(np.float16)
                    except Exception as e:
                        result['errors'].append(f"flow: {str(e)[:50]}")
                        
                        encoded['motion_encoded'] = np.zeros(
                            (1, 256, self.num_frames, 128, 128), 
                            dtype=np.float16
                        )
                else:
                   
                    encoded['motion_encoded'] = np.zeros(
                        (1, 256, self.num_frames, 128, 128), 
                        dtype=np.float16
                    )
                
             
                if 'reference_frame' in data:
                    try:
                        tensor = self._prepare_reference(data['reference_frame'])
                        enc = self.encoders['style'](tensor, num_frames=self.num_frames)
                        encoded['style_encoded'] = enc.cpu().numpy().astype(np.float16)
                    except Exception as e:
                        result['errors'].append(f"style: {str(e)[:50]}")
                
                if 'pose_sequence' in data:
                    try:
                        tensor = self._prepare_pose(data['pose_sequence'])
                        enc = self.encoders['pose'](tensor)
                        encoded['pose_encoded'] = enc.cpu().numpy().astype(np.float16)
                    except Exception as e:
                        result['errors'].append(f"pose: {str(e)[:50]}")
                       
                        encoded['pose_encoded'] = np.zeros(
                            (1, 256, self.num_frames, 128, 128), 
                            dtype=np.float16
                        )
                else:
                    
                    encoded['pose_encoded'] = np.zeros(
                        (1, 256, self.num_frames, 128, 128), 
                        dtype=np.float16
                    )
            
               
                if 'masks' in data:
                    try:
                        tensor = self._prepare_mask(data['masks'])
                        enc = self.encoders['mask'](tensor)
                        encoded['mask_encoded'] = enc.cpu().numpy().astype(np.float16)
                    except Exception as e:
                        result['errors'].append(f"mask: {str(e)[:50]}")
                
               
                
               
                if len(encoded) >= 4:  
                    np.savez_compressed(file_info['output_path'], **encoded)
                    result['success'] = True
                    result['encodings'] = len(encoded)
                    result['size_mb'] = file_info['output_path'].stat().st_size / 1e6
            
        except Exception as e:
            result['errors'].append(f"general: {str(e)[:100]}")
        
        return result
    
    def process_all(self):
        """Process all files"""
        print("="*70)
        print(f"Processing {len(self.npz_files)} files")
        print(f"Config: {self.num_frames} frames @ {self.resolution}")
        print("="*70 + "\n")
        
        success_count = 0
        total_size_mb = 0
        total_encodings = 0
        all_errors = []
        
        start_time = time.time()
        
        for i, file_info in enumerate(tqdm(self.npz_files, desc="Processing", ncols=80)):
            result = self.process_single_file(file_info)
            
            if result['success']:
                success_count += 1
                total_size_mb += result['size_mb']
                total_encodings += result['encodings']
            
            if result['errors']:
                all_errors.extend(result['errors'])
            
            # Clear cache periodically
            if (i + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        elapsed = time.time() - start_time
        
        
        print(f"\n{'='*70}")
        print("Processing Complete!")
        print("="*70)
        print(f"Success: {success_count}/{len(self.npz_files)} files")
        print(f"Total encodings: {total_encodings}")
        print(f"Total size: {total_size_mb/1024:.2f} GB")
        print(f"Time elapsed: {elapsed/60:.1f} minutes")
        print(f"Speed: {len(self.npz_files)/elapsed:.1f} files/second")
        
        if all_errors:
            print(f"\nErrors encountered: {len(all_errors)}")
            print("First 10 errors:")
            for err in all_errors[:10]:
                print(f"  - {err}")
        
        print("="*70 + "\n")
        
        self.print_stats()
    
    def print_stats(self):
        """Print statistics"""
        processed = list(self.output_dir.rglob('*_encoded.npz'))
        
        if not processed:
            print("No files found")
            return
        
        print("Sample File Contents:")
        print("-"*70)
        
        sample = np.load(processed[0])
        for key in sample.keys():
            data = sample[key]
            print(f"{key:20s}: {str(data.shape):35s} {data.nbytes/1e6:8.2f} MB")
        
        print("-"*70)
        print(f"Total files: {len(processed)}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--control_dir', type=str, 
                       default='data/control_signals',
                       help='Base directory containing control NPZ files')
    parser.add_argument('--output_dir', type=str,
                       default='data/encoded_controls',
                       help='Output directory for encoded features')
    parser.add_argument('--num_frames', type=int, default=8,
                       help='Number of frames to sample')
    parser.add_argument('--resolution', type=int, nargs=2, default=(256, 256),
                       help='Target resolution (W H)')
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    processor = ControlEncoderProcessor(
        control_base_dir=args.control_dir,
        output_dir=args.output_dir,
        device=device,
        num_frames=args.num_frames,
        resolution=tuple(args.resolution)
    )
    
    processor.process_all()


if __name__ == '__main__':
    main()