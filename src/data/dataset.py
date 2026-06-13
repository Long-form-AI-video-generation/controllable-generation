

import sys
import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import cv2
from typing import Dict

# Reuse the adapter's canonical modality order so the positional decoding of the
# valid_modalities array can never drift from the order it was written in.
sys.path.append(str(Path(__file__).resolve().parent.parent))  # src/
from models.control_adapter import MODALITY_ORDER, SPATIAL_MODALITIES



class ControllableVideoDataset(Dataset):
   
    def __init__(
        self,
        encoded_controls_dir: str,
        videos_dir: str,
        annotations_path: str,
        num_frames: int = 8,
        resolution: tuple = (128, 128),
        split: str = 'train',
        text_encoder=None,
        load_videos: bool = True
    ):
        """
        Args:
            encoded_controls_dir: Base directory with encoded controls
            videos_dir: Directory with video files
            annotations_path: Path to shots_metadata.json
            num_frames: Number of frames to sample
            resolution: Target resolution (W, H)
            split: 'train' (60%), 'val' (20%), or 'test' (20%)
            load_videos: If False, return dummy frames (for testing)
        """
        self.encoded_dir = Path(encoded_controls_dir)
        self.videos_dir = Path(videos_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.split = split
        self.load_videos = load_videos
        
        print(f"\n{'='*70}")
        print(f"Loading mutlt-Video Dataset - {split.upper()} split")
        print(f"{'='*70}")
        
        
        print("  Loading annotations...")
        with open(annotations_path) as f:
            all_annotations = json.load(f)
        
        print(f"  Total shots: {len(all_annotations)}")
        
        
        all_annotations = sorted(all_annotations, key=lambda x: x['shot_id'])
        
        
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
        
        print(f"  {split.capitalize()} shots: {len(split_annotations)}")
        
        
        self.ann_lookup = {}
        for ann in split_annotations:
           
            video_id = ann['video_id']   
            shot_id = ann['shot_id']

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
            print(" No samples found!")
            print(f"  Annotation keys (first 5): {list(self.ann_lookup.keys())[:5]}")
            print("  Checking encoded files...")
            enc_files = list(self.encoded_dir.rglob('*_encoded.npz'))
            print(f"  Found {len(enc_files)} encoded files")
            if enc_files:
                print(f"  Example file: {enc_files[0]}")
                rel = enc_files[0].relative_to(self.encoded_dir)
                print(f"  Example video_id: {rel.parent.name}")
                print(f"  Example shot_id: {rel.stem.replace('_encoded', '')}")

        self.text_cache = {}
        if text_encoder is not None:
            print("  Pre-encoding text embeddings...")
            unique_captions = list({s['caption'] for s in self.samples})
            print(f"  Unique captions: {len(unique_captions)} / {len(self.samples)} total")
            
            batch_size = 8
            for i in range(0, len(unique_captions), batch_size):
                batch = unique_captions[i:i+batch_size]
                embeddings = text_encoder.encode_text(batch)
                for caption, emb in zip(batch, embeddings):
                    self.text_cache[caption] = emb.cpu()
            print("  Text encoding complete.")
        else:
           
            for s in self.samples:
                self.text_cache[s['caption']] = s['caption']
    
    def __len__(self):
        return len(self.samples)
    
    def _load_video_frames(self, video_id: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load video frames"""
        if not self.load_videos:
            return torch.zeros(self.num_frames, 3, *self.resolution)
        
        
        video_paths = [
            self.videos_dir / f"{video_id}.mp4",
            self.videos_dir / video_id / "video.mp4",
            self.videos_dir / video_id / f"{video_id}.mp4",
        ]
        
        video_path = None
        for path in video_paths:
            if path.exists():
                video_path = path
                break
        
        if video_path is None:
            print(f"⚠️  Video not found for {video_id}")
            print(f"   Tried: {[str(p) for p in video_paths]}")
            return torch.zeros(self.num_frames, 3, *self.resolution)
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
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
            frame_indices = np.linspace(start_frame, end_frame - 1, self.num_frames, dtype=int)
        
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

    # Canonical modality order imported from control_adapter (single source of truth).
    # Spatial modalities are encoded as (256, T, H, W) volumes; style is a (768,) CLIP
    # embedding handled separately.
    MODALITY_ORDER = MODALITY_ORDER
    SPATIAL_MODALITIES = SPATIAL_MODALITIES
    STYLE_DIM = 768

    def _fill_and_validate(self, controls: dict, encoded) -> tuple:
        """Guarantee all 6 canonical keys exist and return (controls, valid).

        `valid[key]` is a scalar tensor (1. present, 0. absent/zero-filled). Uses the
        authoritative `valid_modalities` array written by encode_controls.py when
        present, otherwise falls back to nonzero detection (older encoded files).
        """
        flags = encoded['valid_modalities'] if 'valid_modalities' in encoded else None
        if flags is not None and len(flags) != len(self.MODALITY_ORDER):
            # On-disk array no longer matches the canonical layout — decoding it
            # positionally would mask the wrong modality. Fail loudly instead.
            raise ValueError(
                f"valid_modalities has {len(flags)} entries but MODALITY_ORDER has "
                f"{len(self.MODALITY_ORDER)} ({self.MODALITY_ORDER}); re-encode the data."
            )

        # Shape to zero-fill missing spatial modalities: copy a present one, else default.
        spatial_shape = None
        for mod in self.SPATIAL_MODALITIES:
            key = f'{mod}_encoded'
            if key in controls:
                spatial_shape = tuple(controls[key].shape)
                break
        if spatial_shape is None:
            spatial_shape = (256, self.num_frames, 128, 128)

        valid = {}
        for i, mod in enumerate(self.MODALITY_ORDER):
            key = f'{mod}_encoded'
            if mod == 'style':
                if key not in controls:
                    controls[key] = torch.zeros(self.STYLE_DIM)
            else:
                if key not in controls:
                    controls[key] = torch.zeros(spatial_shape)

            if flags is not None:
                present = float(flags[i])
            else:
                present = float(controls[key].abs().sum() > 0)
            valid[key] = torch.tensor(present, dtype=torch.float32)

        return controls, valid

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        
        try:
            # Load encoded controls
            encoded = np.load(sample['encoded_path'])

            controls = {}
            for key in encoded.keys():
                if key == 'valid_modalities':
                    continue  # handled separately below
                data = encoded[key]
                tensor = torch.from_numpy(data).float()

                if key == 'style_encoded':
                    # BUG 3: style is a (1, 768) CLIP embedding -> store as (768,).
                    tensor = tensor.reshape(-1)
                elif tensor.dim() == 5 and tensor.shape[0] == 1:
                    tensor = tensor.squeeze(0)

                controls[key] = tensor

            # BUG 1: guarantee all 6 canonical modalities exist and emit per-modality
            # validity flags. Missing spatial modalities are zero-filled and marked
            # invalid; the adapter masks them to exactly zero before gating.
            controls, valid = self._fill_and_validate(controls, encoded)

            # Load video frames
            frames = self._load_video_frames(
                sample['video_id'],
                sample['start_frame'],
                sample['end_frame']
            )


            video = frames.permute(1, 0, 2, 3)  # [T, C, H, W] -> [C, T, H, W]

            return {
                'controls': controls,
                'valid': valid,
                'video': video,
                # 'caption': sample['caption'],
                'caption': self.text_cache[sample['caption']],
                'video_id': sample['video_id'],
                'shot_id': sample['shot_id']
            }
        
        except Exception as e:
            print(f"⚠️  Error loading sample {idx}: {e}")
            return {
                'controls': {
                    'depth_encoded': torch.zeros(256, 8, 128, 128),
                    'sketch_encoded': torch.zeros(256, 8, 128, 128),
                    'motion_encoded': torch.zeros(256, 8, 128, 128),
                    'style_encoded': torch.zeros(self.STYLE_DIM),
                    'pose_encoded': torch.zeros(256, 8, 128, 128),
                    'mask_encoded': torch.zeros(256, 8, 128, 128),
                },
                # All modalities invalid: a dummy sample contributes no control gradient.
                'valid': {f'{m}_encoded': torch.tensor(0.0, dtype=torch.float32)
                          for m in self.MODALITY_ORDER},
                'video': torch.zeros(3, self.num_frames, *self.resolution),
                'caption': "error loading sample",
                'video_id': 'error',
                'shot_id': 'error'
            }
def test_dataset():
    """Test dataset loading"""
    dataset = ControllableVideoDataset(
        encoded_controls_dir='/mnt/d1/controllable-generation/encoded_controls',
        videos_dir='/mnt/d1/controllable-generation/videos',
        annotations_path='/mnt/d1/controllable-generation/shots_metadata.json',
        split='train',
        load_videos=False
    )
    
    print(f"\nDataset size: {len(dataset)}")
    
    if len(dataset) > 0:
        sample = dataset[100]
        print(f"\nSample structure:")
        print(f"  Controls: {list(sample['controls'].keys())}")
        for k, v in sample['controls'].items():
            print(f"    {k}: {v.shape}")
        
        print(f"  Caption: {sample['caption'][:60]}...")


if __name__ == '__main__':
    test_dataset()
