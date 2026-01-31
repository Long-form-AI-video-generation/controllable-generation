import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import cv2
from typing import Dict, Optional


class ControllableVideoDataset(Dataset):
    
    def __init__(
        self,
        encoded_controls_dir: str,
        videos_dir: str,
        annotations_path: str,
        num_frames: int = 8,
        resolution: tuple = (512, 512),
        split: str = 'train',
        load_videos: bool = True
    ):
        """
        Args:
            encoded_controls_dir: Directory with *_encoded.npz files
            videos_dir: Directory with video files
            annotations_path: Path to shots_metadata.json
            num_frames: Number of frames to sample
            resolution: Target resolution (W, H)
            split: 'train', 'val', or 'test'
            load_videos: If False, return dummy frames (for testing)
        """
        self.encoded_dir = Path(encoded_controls_dir)
        self.videos_dir = Path(videos_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.split = split
        self.load_videos = load_videos
        
        print(f"\n{'='*70}")
        print(f"Loading Dataset - {split.upper()} split")
        print(f"{'='*70}")
        
        print("  Loading annotations...")
        with open(annotations_path) as f:
            all_annotations = json.load(f)
        
      
        videos_with_shots = {}
        for ann in all_annotations:
            video_id = ann['video_id']
            if video_id not in videos_with_shots:
                videos_with_shots[video_id] = []
            videos_with_shots[video_id].append(ann)
      
        all_video_ids = sorted(videos_with_shots.keys())
        
        # Split by videos (not shots) to avoid data leakage
        # 70% train, 20% val, 10% test
        num_videos = len(all_video_ids)
        train_end = int(num_videos * 0.7)
        val_end = int(num_videos * 0.9)
        
        if split == 'train':
            video_ids = all_video_ids[:train_end]
        elif split == 'val':
            video_ids = all_video_ids[train_end:val_end]
        elif split == 'test':
            video_ids = all_video_ids[val_end:]
        else:
            raise ValueError(f"Invalid split: {split}")
        
        # Get annotations for this split
        split_annotations = []
        for video_id in video_ids:
            split_annotations.extend(videos_with_shots[video_id])
        
        print(f"  Total videos: {num_videos}")
        print(f"  Split videos: {len(video_ids)}")
        print(f"  Split shots: {len(split_annotations)}")
        
     
        self.ann_lookup = {}
        for ann in split_annotations:
            key = f"{ann['video_id']}_{ann['shot_id']}"
            self.ann_lookup[key] = ann
        
      
        print("  Finding encoded files...")
        self.samples = []
        
        for enc_file in self.encoded_dir.rglob('*_encoded.npz'):
          
            rel_path = enc_file.relative_to(self.encoded_dir)
            video_id = rel_path.parent.name
            
            
            if video_id not in video_ids:
                continue
            
            shot_id = rel_path.stem.replace('_encoded', '')
            
            
            key = f"{video_id}_{shot_id}"
            if key in self.ann_lookup:
                ann = self.ann_lookup[key]
                
                caption = ann.get('narrative_caption', '')
                if not caption:
                    caption = ann.get('descriptive_caption', '')
                if not caption:
                    caption = f"Video {video_id} shot {shot_id}"
                
                self.samples.append({
                    'encoded_path': enc_file,
                    'video_id': video_id,
                    'shot_id': shot_id,
                    'caption': caption,
                    'start_frame': ann['segment_start_frame'],
                    'end_frame': ann['segment_end_frame'],
                    'fps': ann.get('fps', 30.0)
                })
        
        print(f"  Valid samples: {len(self.samples)}")
        print(f"{'='*70}\n")
        
        if len(self.samples) == 0:
            print("⚠️  WARNING: No samples found!")
            print("  Check that:")
            print("  1. encoded_controls_dir has *_encoded.npz files")
            print("  2. File naming matches: shot_VIDEOID_shot_XXXX_encoded.npz")
            print("  3. Annotations match the file structure")
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(
        self, 
        video_id: str, 
        start_frame: int, 
        end_frame: int
    ) -> torch.Tensor:
        """
        Load video frames for a shot
        
        Returns:
            frames: [T, C, H, W] tensor, range [0, 1]
        """
        if not self.load_videos:
            # Return dummy frames for testing
            return torch.zeros(self.num_frames, 3, *self.resolution)
        
        video_path = self.videos_dir / f"{video_id}.mp4"
        
        if not video_path.exists():
            print(f"⚠️  Video not found: {video_path}, using black frames")
            return torch.zeros(self.num_frames, 3, *self.resolution)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"⚠️  Cannot open video: {video_path}, using black frames")
            return torch.zeros(self.num_frames, 3, *self.resolution)
        
      
        total_frames = end_frame - start_frame
        
        if total_frames <= 0:
            cap.release()
            return torch.zeros(self.num_frames, 3, *self.resolution)
        
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
        
        return frames
    
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
                
                controls[key] = tensor
           
            frames = self._load_video_frames(
                sample['video_id'],
                sample['start_frame'],
                sample['end_frame']
            )
            
            return {
                'controls': controls,
                'frames': frames,
                'caption': sample['caption'],
                'video_id': sample['video_id'],
                'shot_id': sample['shot_id']
            }
        
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            # Return dummy data
            return {
                'controls': {
                    'depth_encoded': torch.zeros(256, 8, 64, 64),
                    'sketch_encoded': torch.zeros(256, 8, 64, 64),
                    'motion_encoded': torch.zeros(256, 8, 64, 64),
                    'style_encoded': torch.zeros(256, 8, 64, 64),
                    'pose_encoded': torch.zeros(256, 8, 64, 64),
                    'mask_encoded': torch.zeros(256, 8, 64, 64),
                },
                'frames': torch.zeros(8, 3, *self.resolution),
                'caption': "error loading sample",
                'video_id': 'error',
                'shot_id': 'error'
            }


def test_dataset():
    """Test dataset loading"""
    dataset = ControllableVideoDataset(
        encoded_controls_dir='data/encoded_controls',
        videos_dir='data/videos',
        annotations_path='data/shots_metadata.json',
        split='train',
        load_videos=False  # Use dummy frames for quick test
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"\nSample structure:")
        print(f"  Controls: {list(sample['controls'].keys())}")
        for k, v in sample['controls'].items():
            print(f"    {k}: {v.shape}")
        print(f"  Frames: {sample['frames'].shape}")
        print(f"  Caption: {sample['caption'][:60]}...")


if __name__ == '__main__':
    test_dataset()