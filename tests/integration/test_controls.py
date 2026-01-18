
"""
Quick test to verify pre-extracted controls load correctly.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from data.dataset import AnimeControlDataset


def quick_test():
    """Quick test with your exact file structure."""
    
    print("Quick test of pre-extracted controls...\n")
    
    
    dataset = AnimeControlDataset(
        annotation_file='../../data/shots_metadata.json',
        video_dir='../../data/videos',
        control_dir='../../data/control_signals',
        num_frames=16,
        resolution=256,
        control_types=['depth', 'edges', 'flow'],
        use_preextracted=True
    )
    
    print(f"Dataset size: {len(dataset)} shots\n")
    
    if len(dataset) == 0:
        print("❌ No shots found!")
        print("\nDebugging info:")
        
        
        control_dir = Path('../../data/control_signals')
        print(f"Control directory exists: {control_dir.exists()}")
        print(f"Video IDs in control dir:")
        for d in sorted(control_dir.iterdir()):
            if d.is_dir():
                num_files = len(list(d.glob('*.npz')))
                print(f"  {d.name}: {num_files} .npz files")
        
        # Check annotation file
        import json
        with open('../test_data/test_shots.json', 'r') as f:
            data = json.load(f)
            shots = data['shots'] if isinstance(data, dict) else data
            print(f"\nShots in annotation file: {len(shots)}")
            print(f"First 3 video IDs:")
            for shot in shots[:3]:
                video_id = shot['video_id']
                shot_id = shot['shot_id']
                control_file = control_dir / video_id / f"shot_{shot_id}_controls.npz"
                print(f"  {video_id} / {shot_id}")
                print(f"    Expected: {control_file}")
                print(f"    Exists: {control_file.exists()}")
        
        return
    
    # Load first sample
    print("Loading first sample...")
    sample = dataset[0]
    
    print(f"\n✓ Successfully loaded sample!")
    print(f"  Shot ID: {sample['shot_id']}")
    print(f"  Caption: {sample['caption'][:60]}...")
    print(f"\nControl shapes:")
    print(f"  depth: {sample['depth'].shape}")
    print(f"  edges: {sample['edges'].shape}")
    print(f"  flow: {sample['flow'].shape}")
    print(f"  reference_frame: {sample['reference_frame'].shape}")
    print(f"  style_embedding: {sample['style_embedding'].shape}")
    print(f"  mask: {sample['mask'].shape}")
    
    print(f"\n✅ Quick test PASSED!")
    print(f"\nYou have {len(dataset)} shots ready for training!")


if __name__ == '__main__':
    quick_test()