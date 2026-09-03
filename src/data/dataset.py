

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
import json
import cv2
from typing import Dict

try:
    from data.frame_sampling import (
        resolve_frame_interval,
        select_frame_indices,
    )
except ImportError:  # Support package imports from the repository root.
    from src.data.frame_sampling import (
        resolve_frame_interval,
        select_frame_indices,
    )



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
        load_videos: bool = True,
        control_key: str = 'depth_encoded',
        strict: bool = False,
        split_manifest_path: str = None,
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
            control_key: The single control tensor returned for each sample
            strict: Raise data errors instead of returning fallback samples
            split_manifest_path: Optional frozen train/val/test video-ID manifest
        """
        self.encoded_dir = Path(encoded_controls_dir)
        self.videos_dir = Path(videos_dir)
        self.num_frames = num_frames
        self.resolution = resolution
        self.split = split
        self.load_videos = load_videos
        self.control_key = control_key
        self.strict = strict
        self.split_manifest_path = (
            Path(split_manifest_path) if split_manifest_path else None
        )

        if not self.control_key:
            raise ValueError("control_key must not be empty")
        
        print(f"\n{'='*70}")
        print(f"Loading mutlt-Video Dataset - {split.upper()} split")
        print(f"{'='*70}")
        
        
        print("  Loading annotations...")
        with open(annotations_path) as f:
            all_annotations = json.load(f)
        
        print(f"  Total shots: {len(all_annotations)}")
        
        
        all_annotations = sorted(all_annotations, key=lambda x: x['shot_id'])
        
        
        if split not in {'train', 'val', 'test'}:
            raise ValueError(f"Invalid split: {split}")

        if self.split_manifest_path is not None:
            split_annotations = self._select_manifest_annotations(
                all_annotations,
                split,
            )
        else:
            num_shots = len(all_annotations)
            train_end = int(num_shots * 0.6)
            val_end = int(num_shots * 0.8)
            if split == 'train':
                split_annotations = all_annotations[:train_end]
            elif split == 'val':
                split_annotations = all_annotations[train_end:val_end]
            else:
                split_annotations = all_annotations[val_end:]
        
        print(f"  {split.capitalize()} shots: {len(split_annotations)}")
        
        
        self.ann_lookup = {}
        for ann in split_annotations:
           
            video_id = ann['video_id']   
            shot_id = ann['shot_id']

            key = f"{video_id}_{shot_id}"
            self.ann_lookup[key] = ann

        
        
        print("  Finding encoded files...")
        self.samples = []
        
        for enc_file in sorted(self.encoded_dir.rglob('*_encoded.npz')):
          
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

    def _select_manifest_annotations(self, all_annotations, split):
        with open(self.split_manifest_path) as manifest_file:
            manifest = json.load(manifest_file)

        required = {'train', 'val', 'test'}
        if not required.issubset(manifest):
            raise ValueError(
                f"split manifest must contain {sorted(required)}"
            )

        split_sets = {
            name: {str(video_id) for video_id in manifest[name]}
            for name in required
        }
        if (
            split_sets['train'] & split_sets['val']
            or split_sets['train'] & split_sets['test']
            or split_sets['val'] & split_sets['test']
        ):
            raise ValueError("split manifest contains overlapping video IDs")

        target_ids = split_sets[split]
        selected = [
            annotation
            for annotation in all_annotations
            if str(annotation['video_id']) in target_ids
        ]
        found_ids = {str(annotation['video_id']) for annotation in selected}
        missing = target_ids - found_ids
        if missing:
            preview = sorted(missing)[:10]
            raise ValueError(
                f"split manifest references missing video IDs: {preview}"
            )
        return selected
    
    def __len__(self):
        return len(self.samples)

    def _empty_video_frames(self) -> torch.Tensor:
        return torch.zeros(
            self.num_frames,
            3,
            self.resolution[1],
            self.resolution[0],
        )
    
    def _load_video_frames(self, video_id: str, start_frame: int, end_frame: int) -> torch.Tensor:
        """Load video frames"""
        if not self.load_videos:
            return self._empty_video_frames()
        
        
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
            if self.strict:
                raise FileNotFoundError(
                    f"Video not found for {video_id}; tried {video_paths}"
                )
            print(f"⚠️  Video not found for {video_id}")
            print(f"   Tried: {[str(p) for p in video_paths]}")
            return self._empty_video_frames()
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            if self.strict:
                raise RuntimeError(f"Could not open video: {video_path}")
            return self._empty_video_frames()
        
        actual_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        try:
            start_frame, end_frame = resolve_frame_interval(
                start_frame,
                end_frame,
                actual_frame_count,
            )
        except ValueError as error:
            cap.release()
            if self.strict:
                raise ValueError(
                    f"Invalid frame range for {video_path}: {error}"
                )
            return self._empty_video_frames()

        frame_indices = select_frame_indices(
            start_frame,
            end_frame,
            self.num_frames,
        )
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                if self.strict:
                    cap.release()
                    raise RuntimeError(
                        f"Could not decode frame {frame_idx} from {video_path}"
                    )
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
    
    

    def _load_control(self, encoded_path: Path) -> torch.Tensor:
        with np.load(encoded_path, allow_pickle=False) as encoded:
            if self.control_key not in encoded:
                raise KeyError(
                    f"{encoded_path} does not contain {self.control_key!r}"
                )
            data = np.asarray(encoded[self.control_key])

        if self.control_key == 'mask_encoded':
            if not np.issubdtype(data.dtype, np.integer):
                raise TypeError(f"{encoded_path}: mask must contain integer IDs")
            tensor = torch.from_numpy(data.astype(np.int64, copy=False))
        else:
            tensor = torch.from_numpy(data).half()
        if tensor.dim() == 5 and tensor.shape[0] == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() != 4:
            raise ValueError(
                f"{encoded_path}: expected [C,T,H,W], "
                f"got {tuple(tensor.shape)}"
            )
        if (
            tensor.shape[1] != self.num_frames
            and (self.strict or self.control_key in {'sketch_encoded', 'mask_encoded'})
        ):
            raise ValueError(
                f"{encoded_path}: expected {self.num_frames} control frames, "
                f"got {tensor.shape[1]}"
            )
        if tensor.is_floating_point() and not torch.isfinite(tensor).all():
            raise ValueError(f"{encoded_path}: control contains NaN or Inf")

        if self.control_key == 'mask_encoded':
            if tensor.shape[0] != 1:
                raise ValueError(f"{encoded_path}: mask must have one channel")
            expected_spatial = (self.resolution[1], self.resolution[0])
            if tensor.shape[-2:] != expected_spatial:
                raise ValueError(f"{encoded_path}: mask spatial shape must be {expected_spatial}")
            if tensor.numel() == 0 or int(tensor.min()) < 0 or int(tensor.max()) > 149:
                raise ValueError(f"{encoded_path}: mask IDs must be in [0,149]")

        if self.control_key == 'sketch_encoded':
            if tensor.shape[0] != 1:
                raise ValueError(
                    f"{encoded_path}: sketch must have one channel, "
                    f"got {tensor.shape[0]}"
                )
            expected_spatial = (self.resolution[1], self.resolution[0])
            if tensor.shape[-2:] != expected_spatial:
                raise ValueError(
                    f"{encoded_path}: sketch spatial shape must be "
                    f"{expected_spatial}, got {tuple(tensor.shape[-2:])}"
                )
            low = float(tensor.min())
            high = float(tensor.max())
            if low < 0.0 or high > 1.0:
                raise ValueError(
                    f"{encoded_path}: sketch range must be [0,1], "
                    f"got [{low}, {high}]"
                )

        return tensor

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        try:
            controls = {
                self.control_key: self._load_control(sample['encoded_path'])
            }
            frames = self._load_video_frames(
                sample['video_id'],
                sample['start_frame'],
                sample['end_frame'],
            )
            video = frames.permute(1, 0, 2, 3)

            return {
                'controls': controls,
                'video': video,
                'caption': self.text_cache[sample['caption']],
                'video_id': sample['video_id'],
                'shot_id': sample['shot_id'],
            }

        except Exception as error:
            if self.strict:
                raise RuntimeError(
                    f"Failed to load sample {idx} "
                    f"({sample['video_id']}/{sample['shot_id']}): {error}"
                ) from error

            print(f"Warning: error loading sample {idx}: {error}")
            control_channels = (
                1 if self.control_key in {'sketch_encoded', 'mask_encoded'} else 256
            )
            return {
                'controls': {
                    self.control_key: torch.zeros(
                        control_channels,
                        self.num_frames,
                        self.resolution[1],
                        self.resolution[0],
                    ),
                },
                'video': torch.zeros(
                    3,
                    self.num_frames,
                    self.resolution[1],
                    self.resolution[0],
                ),
                'caption': "error loading sample",
                'video_id': 'error',
                'shot_id': 'error',
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
