# data/simple_dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
from typing import Dict, Optional

class SimpleControlDataset(Dataset):
  
    
    def __init__(
        self,
        encoded_dir: str,
        annotations_path: str,
        expected_controls: int = 6
    ):
        self.encoded_dir = Path(encoded_dir)
        self.expected_controls = expected_controls
        
     
        print(f"Loading annotations from {annotations_path}...")
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        
      
        print(f"Scanning {encoded_dir} for encoded files...")
        self.samples = self._build_index()
        
        print(f"\n{'='*70}")
        print("Dataset Initialized")
        print(f"{'='*70}")
        print(f"  Total samples: {len(self.samples)}")
        print(f"  Expected controls per sample: {expected_controls}")
        print(f"{'='*70}\n")
    
    def _build_index(self):
        """Build index of all valid samples"""
        samples = []
        
        for npz_path in self.encoded_dir.rglob('*_encoded.npz'):
            rel_path = npz_path.relative_to(self.encoded_dir)
            video_id = rel_path.parent.name
            shot_id = rel_path.stem.replace('_encoded', '')
            
            # Get caption
            caption = self._get_caption(video_id, shot_id)
            
            samples.append({
                'path': npz_path,
                'video_id': video_id,
                'shot_id': shot_id,
                'caption': caption
            })
        
        return samples
    
    def _get_caption(self, video_id: str, shot_id: str) -> str:
        """Get caption from annotations"""
        try:
            if video_id in self.annotations:
                for shot in self.annotations[video_id].get('shots', []):
                    if shot.get('shot_id') == shot_id:
                        return shot.get('caption', 'A video clip')
        except Exception as e:
            pass
        
        return "A video clip"  # Default fallback
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load NPZ file
        try:
            data = np.load(sample['path'])
            

            controls = {}
            for key in data.keys():
                arr = data[key]
               
                if arr.dtype == np.float16:
                    arr = arr.astype(np.float32)
                controls[key] = torch.from_numpy(arr)
           
            if len(controls) != self.expected_controls:
                print(f"Warning: {sample['path']} has {len(controls)} controls, expected {self.expected_controls}")
            
            return {
                'controls': controls,
                'caption': sample['caption'],
                'video_id': sample['video_id'],
                'shot_id': sample['shot_id']
            }
            
        except Exception as e:
            print(f"Error loading {sample['path']}: {e}")
         
            return {
                'controls': self._get_dummy_controls(),
                'caption': "Error loading",
                'video_id': sample['video_id'],
                'shot_id': sample['shot_id']
            }
    
    def _get_dummy_controls(self) -> Dict[str, torch.Tensor]:
        """Create dummy controls for error cases"""
        B, C, T, H, W = 1, 256, 8, 64, 64
        return {
            f'{name}_encoded': torch.zeros(B, C, T, H, W)
            for name in ['depth', 'mask', 'motion', 'pose', 'sketch', 'style']
        }


if __name__ == '__main__':
    # Test the dataset
    dataset = SimpleControlDataset(
        encoded_dir='data/encoded_controls',
        annotations_path='data/annotations.json'
    )
    
    print(f"\nTesting dataset...")
    sample = dataset[0]
    print(f"\nSample 0:")
    print(f"  Video ID: {sample['video_id']}")
    print(f"  Shot ID: {sample['shot_id']}")
    print(f"  Caption: {sample['caption']}")
    print(f"  Controls: {list(sample['controls'].keys())}")
    
    for name, tensor in sample['controls'].items():
        print(f"    {name}: {tensor.shape}")