import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from models.encode_controls import ControlEncoderProcessor


def encode_single_video(video_id: str):
    
    
    print("="*70)
    print(f"Encoding Single Video: {video_id}")
    print("="*70)
    
   
    control_base = Path('data/control_signals') / video_id
    output_base = Path('data/test_single_video/encoded_controls')
    
    if not control_base.exists():
        print(f"\n❌ Raw controls not found: {control_base}")
        print("\nDid you run extract_controls.py?")
        return
    
    raw_files = list(control_base.glob('*.npz'))
    print(f"\n✓ Found {len(raw_files)} raw control files")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    processor = ControlEncoderProcessor(
        control_base_dir=str(control_base),
        output_dir=str(output_base / video_id),
        device=device,
        num_frames=8,
        resolution=(256, 256)
    )
    
    
    processor.process_all()
   
    encoded_files = list((output_base / video_id).rglob('*_encoded.npz'))
    print(f"\n✅ Created {len(encoded_files)} encoded files")
    print(f"   Location: {output_base / video_id}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_id', type=str, default='--GVEgZn_TI',
                       help='Video ID to test')
    args = parser.parse_args()
    
    encode_single_video(args.video_id)