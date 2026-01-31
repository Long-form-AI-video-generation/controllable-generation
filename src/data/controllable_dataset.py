
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class ControllableVideoDataset(Dataset):
    def __init__(
        self,
        encoded_controls_dir: str,
        videos_dir: str,
        annotations_path: str,
        num_frames: int = 8
    ):
        self.encoded_dir = Path(encoded_controls_dir)
        self.videos_dir = Path(videos_dir)
        
        
        import json
        with open(annotations_path) as f:
            self.annotations = json.load(f)
        
      
        self.samples = []
        for enc_file in self.encoded_dir.rglob('*_encoded.npz'):
            rel_path = enc_file.relative_to(self.encoded_dir)
            video_id = rel_path.parent.name
            shot_id = rel_path.stem.replace('_encoded', '')
            
           
            ann = self._find_annotation(video_id, shot_id)
            if ann is not None:
                self.samples.append({
                    'encoded_path': enc_file,
                    'video_id': video_id,
                    'shot_id': shot_id,
                    'caption': ann['caption'],
                    'start_time': ann['start_time'],
                    'end_time': ann['end_time']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
      
        encoded = np.load(sample['encoded_path'])
        controls = {
            key: torch.from_numpy(encoded[key]).float()
            for key in encoded.keys()
        }
        
       
        frames = self._load_video_frames(
            sample['video_id'],
            sample['start_time'],
            sample['end_time']
        )
        
        return {
            'controls': controls,
            'frames': frames,
            'caption': sample['caption']
        }