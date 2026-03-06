import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from torch.utils.data import DataLoader
from single_video_dataset import SingleVideoDataset  # ✅ Use SingleVideoDataset


def train_single_video(video_id: str = '--GVEgZn_TI'):
   
    
    print("="*70)
    print(f"Training Test - Single Video: {video_id}")
    print("="*70)
    
  
    encoded_base = Path('data/test_single_video/encoded_controls')
    encoded_dir = encoded_base / video_id
    
    if not encoded_dir.exists():
        print(f"\n❌ Encoded directory not found: {encoded_dir}")
        print("\nPlease run first:")
        print(f"  python scripts/encode_single_video.py --video_id {video_id}")
        return
    
    encoded_files = list(encoded_dir.rglob('*_encoded.npz'))
    print(f"\n✓ Found {len(encoded_files)} encoded files in: {encoded_dir}")
    
    if len(encoded_files) == 0:
        print("\n❌ No encoded files found!")
        return
   
    print("\nSample encoded files:")
    for f in encoded_files[:3]:
        print(f"  - {f.name}")
   
    print("\n[1/4] Creating annotations from full dataset...")
    import json
    
    full_annotations_path = Path('data/shots_metadata.json')
    if not full_annotations_path.exists():
        print(f"\n❌ Annotations not found: {full_annotations_path}")
        return
    
    with open(full_annotations_path) as f:
        all_annotations = json.load(f)
    
    video_annotations = [
        ann for ann in all_annotations
        if ann['video_id'] == video_id
    ]
    
    print(f"   ✓ Found {len(video_annotations)} annotations for {video_id}")
    
    if len(video_annotations) == 0:
        print(f"\n❌ No annotations found for video: {video_id}")
        print("\nAvailable videos in annotations:")
        videos = set(ann['video_id'] for ann in all_annotations)
        for v in sorted(list(videos))[:10]:
            print(f"  - {v}")
        return
    

    test_dir = Path('data/test_single_video')
    test_dir.mkdir(parents=True, exist_ok=True)
    
    test_ann_path = test_dir / 'shots_metadata.json'
    with open(test_ann_path, 'w') as f:
        json.dump(video_annotations, f, indent=2)
    
    print(f"   ✓ Saved annotations to: {test_ann_path}")
    
  
    if video_annotations:
        sample_ann = video_annotations[0]
        print(f"\n   Sample annotation:")
        print(f"     video_id: {sample_ann['video_id']}")
        print(f"     shot_id: {sample_ann['shot_id']}")
        print(f"     caption: {sample_ann.get('narrative_caption', 'N/A')[:50]}...")
 
    print("\n[2/4] Loading dataset with SingleVideoDataset...")
    
    try:
        dataset = SingleVideoDataset(
            encoded_controls_dir=str(encoded_base),  
            videos_dir='data/videos',
            annotations_path=str(test_ann_path),
            num_frames=8,
            resolution=(256, 256),
            split='train',
            load_videos=False  
        )
    except Exception as e:
        print(f"\n❌ Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"   ✓ Dataset size: {len(dataset)}")
    
    if len(dataset) == 0:
        print("\n❌ No samples found!")
        print("\n🔍 Debug Info:")
        print(f"  Encoded dir: {encoded_base}")
        print(f"  Video subdir: {encoded_dir}")
        print(f"  Encoded files: {len(encoded_files)}")
        print(f"  Annotations: {len(video_annotations)}")
        
        if encoded_files:
            print(f"\n  First encoded file:")
            print(f"    Path: {encoded_files[0]}")
            print(f"    Name: {encoded_files[0].name}")
            print(f"    Stem: {encoded_files[0].stem}")
        
        if video_annotations:
            print(f"\n  First annotation:")
            print(f"    video_id: {video_annotations[0]['video_id']}")
            print(f"    shot_id: {video_annotations[0]['shot_id']}")
        
        print("\n  Expected key format: VIDEO_ID_SHOT_ID")
        if encoded_files and video_annotations:
            enc_stem = encoded_files[0].stem.replace('_encoded', '')
            ann_key = f"{video_annotations[0]['video_id']}_{video_annotations[0]['shot_id']}"
            print(f"    Encoded stem: {enc_stem}")
            print(f"    Annotation key: {ann_key}")
            print(f"    Match: {enc_stem == video_annotations[0]['shot_id']}")
        
        return
    
    print("\n   Testing all splits:")
    for split_name in ['train', 'val', 'test']:
        split_dataset = SingleVideoDataset(
            encoded_controls_dir=str(encoded_base),
            videos_dir='data/videos',
            annotations_path=str(test_ann_path),
            num_frames=8,
            resolution=(256, 256),
            split=split_name,
            load_videos=False
        )
        print(f"     {split_name:5s}: {len(split_dataset):3d} samples")
    
  
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
 
    print("\n[3/4] Testing data loading...")
    batch = next(iter(dataloader))
    
    print(f"   ✓ Batch loaded successfully")
    print(f"   - Controls: {list(batch['controls'].keys())}")
    print(f"   - Frames shape: {batch['frames'].shape}")
    print(f"   - Caption: {batch['caption'][0][:60]}...")
 
    print("\n[4/4] Verifying control shapes...")
    for key, value in batch['controls'].items():
        print(f"   {key:20s}: {value.shape}")
    
    print("\n" + "="*70)
    print("✅ Single-Video Test Complete!")
    print("="*70)
    print(f"\nDataset Statistics:")
    print(f"  Total shots: {len(video_annotations)}")
    print(f"  Train split: ~{int(len(video_annotations) * 0.6)} shots (60%)")
    print(f"  Val split: ~{int(len(video_annotations) * 0.2)} shots (20%)")
    print(f"  Test split: ~{int(len(video_annotations) * 0.2)} shots (20%)")
    print("\nNext steps:")
    print("  1. Enable load_videos=True to load actual frames")
    print("  2. Apply fixes to encode_controls.py and re-encode for all 6 modalities")
    print("  3. Run actual training with WAN model")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', type=str, default='--GVEgZn_TI',
                       help='Video ID to test')
    args = parser.parse_args()
    
    train_single_video(args.video_id)