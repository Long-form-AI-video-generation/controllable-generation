import json
from pathlib import Path
import shutil


def create_single_video_split(
    annotations_path: str,
    video_id: str,
    output_dir: str = 'data/test_single_video'
):
    
    print("="*70)
    print(f"Creating Single-Video Test Dataset")
    print("="*70)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(annotations_path) as f:
        all_annotations = json.load(f)
    
    video_annotations = [
        ann for ann in all_annotations 
        if ann['video_id'] == video_id
    ]
    
    print(f"\nVideo: {video_id}")
    print(f"Total shots: {len(video_annotations)}")
    
    if len(video_annotations) == 0:
        print(f"\n❌ No annotations found for video {video_id}")
        print("\nAvailable videos:")
        videos = set(ann['video_id'] for ann in all_annotations)
        for v in sorted(videos):
            count = sum(1 for ann in all_annotations if ann['video_id'] == v)
            print(f"  {v}: {count} shots")
        return
    
    num_shots = len(video_annotations)
    train_end = int(num_shots * 0.6)
    val_end = int(num_shots * 0.8)
    
    splits = {
        'train': video_annotations[:train_end],
        'val': video_annotations[train_end:val_end],
        'test': video_annotations[val_end:]
    }
    
    print(f"\nSplit breakdown:")
    print(f"  Train: {len(splits['train'])} shots ({len(splits['train'])/num_shots*100:.1f}%)")
    print(f"  Val:   {len(splits['val'])} shots ({len(splits['val'])/num_shots*100:.1f}%)")
    print(f"  Test:  {len(splits['test'])} shots ({len(splits['test'])/num_shots*100:.1f}%)")
    
   
    for split_name, split_data in splits.items():
        output_file = output_dir / f'{split_name}_annotations.json'
        with open(output_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        print(f"\n✓ Saved {split_name}: {output_file}")
    
    full_output = output_dir / 'shots_metadata.json'
    with open(full_output, 'w') as f:
        json.dump(video_annotations, f, indent=2)
    print(f"✓ Saved full: {full_output}")
    
    print(f"\n{'='*70}")
    print("✅ Single-video dataset created!")
    print(f"{'='*70}")
    print(f"\nData location: {output_dir}")
    print(f"\nNext steps:")
    print(f"  1. Extract controls (if not done)")
    print(f"  2. Encode controls: python scripts/encode_single_video.py")
    print(f"  3. Train: python scripts/train_single_video.py")
    
    return output_dir


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, 
                       default='data/shots_metadata.json')
    parser.add_argument('--video_id', type=str, default='--GVEgZn_TI',
                       help='Video ID to test')
    parser.add_argument('--output_dir', type=str,
                       default='data/test_single_video')
    
    args = parser.parse_args()
    
    create_single_video_split(
        args.annotations,
        args.video_id,
        args.output_dir
    )