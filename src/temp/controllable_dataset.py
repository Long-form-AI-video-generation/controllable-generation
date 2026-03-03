import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
from pathlib import Path
import json
import cv2
from typing import Optional, Dict


class ControllableVideoDataset(Dataset):
    def __init__(
        self,
        encoded_controls_dir: str,
        videos_dir: str,
        annotations_path: str,
        num_frames: int = 8,
        resolution: tuple = (64, 64),
        control_resolution: tuple = (16, 16),
        split: str = 'train',
        load_videos: bool = True
    ):
       
        self.encoded_dir = Path(encoded_controls_dir)
        self.videos_dir = Path(videos_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.control_resolution = control_resolution
        self.split = split
        self.load_videos = load_videos
        
        print(f"\n{'='*70}")
        print(f"Loading Dataset - {split.upper()} split")
        print(f"{'='*70}")
        
      
        print("  Loading annotations...")
        with open(annotations_path) as f:
            all_annotations = json.load(f)
        
        print(f"  Total annotations: {len(all_annotations)}")
        
        
        all_annotations = sorted(all_annotations, key=lambda x: (x.get('video_id', ''), x.get('shot_id', '')))
        
        num_shots = len(all_annotations)
        train_end = int(num_shots * 0.6)
        val_end = int(num_shots * 0.8)
        
        if split == 'train':
            split_annotations = all_annotations[:train_end]
        elif split == 'val':
            split_annotations = all_annotations[train_end:val_end]
        elif split == 'test':
            split_annotations = all_annotations[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        print(f"  {split.capitalize()} annotations: {len(split_annotations)}")
        
    
        self.ann_lookup = {}
        for ann in split_annotations:
            video_id = ann.get('video_id', '')
            shot_id = ann.get('shot_id', '')
            key = f"{video_id}_{shot_id}"
            self.ann_lookup[key] = ann
     
        print("  Finding encoded files...")
        self.samples = []
        
        for enc_file in self.encoded_dir.rglob('*_encoded.npz'):
            rel_path = enc_file.relative_to(self.encoded_dir)
            video_id = rel_path.parent.name
            
         
            stem = rel_path.stem
            stem = stem.replace('_controls_encoded', '')
            stem = stem.replace('_encoded', '')
            
            if stem.startswith('shot_'):
                shot_id = stem[5:]  
            else:
                shot_id = stem
            
           
            ann = self._find_annotation(video_id, shot_id)
            if ann is not None:
               
                caption = ann.get('narrative_caption', '')
                if not caption:
                    caption = ann.get('descriptive_caption', '')
                if not caption:
                    caption = ann.get('caption', '')
                if not caption:
                    caption = f"Video {video_id} shot {shot_id}"
                
                self.samples.append({
                    'encoded_path': enc_file,
                    'video_id': video_id,
                    'shot_id': shot_id,
                    'caption': caption,
                    'start_frame': ann.get('segment_start_frame', ann.get('start_frame', 0)),
                    'end_frame': ann.get('segment_end_frame', ann.get('end_frame', 100)),
                    'start_time': ann.get('start_time', 0.0),
                    'end_time': ann.get('end_time', 3.0),
                    'fps': ann.get('fps', 30.0)
                })
        
        print(f"  Valid samples: {len(self.samples)}")
        print(f"{'='*70}\n")
        
        if len(self.samples) == 0:
            print(" WARNING: No samples found!")
            print(f"  Annotation keys (first 5): {list(self.ann_lookup.keys())[:5]}")
            enc_files = list(self.encoded_dir.rglob('*_encoded.npz'))
            print(f"  Found {len(enc_files)} encoded files")
            if enc_files:
                print(f"  Example file: {enc_files[0]}")
    
    def _find_annotation(self, video_id: str, shot_id: str) -> Optional[Dict]:
        """Find annotation for given video_id and shot_id"""
       
        keys_to_try = [
            f"{video_id}_{shot_id}",
            f"{video_id}_shot_{shot_id}",
            shot_id,
            f"shot_{shot_id}"
        ]
        
        for key in keys_to_try:
            if key in self.ann_lookup:
                return self.ann_lookup[key]
        
        
        for ann in self.ann_lookup.values():
            if (ann.get('video_id', '') == video_id and 
                ann.get('shot_id', '') == shot_id):
                return ann
        
        return None
    
    def _load_video_frames(
        self, 
        video_id: str, 
        start_frame: int = None,
        end_frame: int = None,
        start_time: float = None,
        end_time: float = None
    ) -> torch.Tensor:
        
        if not self.load_videos:
          
            return torch.zeros(3, self.num_frames, *self.resolution)
        
      
        video_paths = [
            self.videos_dir / f"{video_id}.mp4",
            self.videos_dir / f"{video_id}.avi",
            self.videos_dir / video_id / "video.mp4",
            self.videos_dir / video_id / f"{video_id}.mp4",
        ]
        
        video_path = None
        for path in video_paths:
            if path.exists():
                video_path = path
                break
        
        if video_path is None:
            print(f" Video not found: {video_id}")
            return torch.zeros(3, self.num_frames, *self.resolution)
        
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f" Could not open video: {video_path}")
            return torch.zeros(3, self.num_frames, *self.resolution)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        
      
        if start_time is not None and end_time is not None:
            start_frame = int(start_time * fps)
            end_frame = int(end_time * fps)
        elif start_frame is None or end_frame is None:
           
            start_frame = 0
            end_frame = int(3.0 * fps)  
       
        total_frames = end_frame - start_frame
        
        if total_frames <= 0:
            cap.release()
            return torch.zeros(3, self.num_frames, *self.resolution)
        
        if total_frames <= self.num_frames:
           
            frame_indices = list(range(start_frame, end_frame))
            while len(frame_indices) < self.num_frames:
                frame_indices.append(frame_indices[-1])
        else:
           
            frame_indices = np.linspace(
                start_frame, 
                end_frame - 1, 
                self.num_frames, 
                dtype=int
            )
        
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
             
                if len(frames) > 0:
                    frames.append(frames[-1].copy())
                else:
                    frames.append(np.zeros((*self.resolution[::-1], 3), dtype=np.uint8))
            else:
                
                frame = cv2.resize(frame, self.resolution)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
      
        frames = np.stack(frames).astype(np.float32) / 255.0
        frames = torch.from_numpy(frames).permute(0, 3, 1, 2) 
        frames = frames.permute(1, 0, 2, 3) 
        
        return frames
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
          
            encoded = np.load(sample['encoded_path'])
            
            controls = {}
            for key in encoded.keys():
                data = encoded[key]
                tensor = torch.from_numpy(data).float()
                
                
                if tensor.dim() == 5 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)
              
                if tensor.dim() == 4:  
                    C, T, H, W = tensor.shape
                    target_H, target_W = self.control_resolution
                    
                    if H > target_H or W > target_W:
                        tensor = tensor.unsqueeze(0)  
                        tensor = F.interpolate(
                            tensor,
                            size=(T, target_H, target_W),
                            mode='trilinear',
                            align_corners=False
                        )
                        tensor = tensor.squeeze(0)  
                
                controls[key] = tensor
            
            
            frames = self._load_video_frames(
                sample['video_id'],
                start_frame=sample.get('start_frame'),
                end_frame=sample.get('end_frame'),
                start_time=sample.get('start_time'),
                end_time=sample.get('end_time')
            )
            
            return {
                'controls': controls,
                'video': frames, 
                'caption': sample['caption'],
                'video_id': sample['video_id'],
                'shot_id': sample['shot_id']
            }
        
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            
            
            H, W = self.control_resolution
            return {
                'controls': {
                    'depth_encoded': torch.zeros(256, self.num_frames, H, W),
                    'sketch_encoded': torch.zeros(256, self.num_frames, H, W),
                    'motion_encoded': torch.zeros(256, self.num_frames, H, W),
                    'style_encoded': torch.zeros(256, self.num_frames, H, W),
                    'pose_encoded': torch.zeros(256, self.num_frames, H, W),
                    'mask_encoded': torch.zeros(256, self.num_frames, H, W),
                },
                'video': torch.zeros(3, self.num_frames, *self.resolution),
                'caption': "error loading sample",
                'video_id': 'error',
                'shot_id': 'error'
            }


def test_dataset():
    
    dataset = ControllableVideoDataset(
        encoded_controls_dir='data/encoded_controls',
        videos_dir='data/videos',
        annotations_path='data/annotations.json',
        num_frames=4,
        resolution=(64, 64),
        control_resolution=(16, 16),
        split='train',
        load_videos=True
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Controls: {list(sample['controls'].keys())}")
        for k, v in sample['controls'].items():
            print(f"    {k}: {v.shape}")
        print(f"  Video: {sample['video'].shape}")
        print(f"  Caption: {sample['caption'][:60]}...")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  Shot ID: {sample['shot_id']}")


if __name__ == '__main__':
    test_dataset()