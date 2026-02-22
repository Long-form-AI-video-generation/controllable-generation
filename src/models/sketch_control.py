import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import sys
import time

class SketchOnlyProcessorOnlyProcessor:
    """Process control NPZ files — sketch only, no encoder (raw MiDaS values)"""

    def __init__(
        self,
        control_base_dir: str,
        output_dir: str,
        num_frames: int = 8,
        resolution: tuple = (256, 256),
    ):
        self.control_base = Path(control_base_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.num_frames = num_frames
        self.resolution = resolution  # (W, H)

        self.npz_files = self._find_all_npz_files()
        print(f"Found {len(self.npz_files)} NPZ files to process\n")

    def _find_all_npz_files(self) -> list:
        files = []
        for npz_path in self.control_base.rglob('*.npz'):
            # Skip already-encoded files if they ended up in source dir
            if '_encoded' in npz_path.stem:
                continue
            rel_path = npz_path.relative_to(self.control_base)
            output_path = self.output_dir / rel_path.parent / f"{rel_path.stem}_encoded.npz"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            files.append({
                'input_path': npz_path,
                'output_path': output_path,
            })
        return files

    def _sample_frames(self, data: np.ndarray, target_frames: int) -> np.ndarray:
        current_frames = data.shape[0]
        if current_frames <= target_frames:
            if current_frames < target_frames:
                pad_width = [(0, target_frames - current_frames)] + [(0, 0)] * (data.ndim - 1)
                data = np.pad(data, pad_width, mode='edge')
            return data
        indices = np.linspace(0, current_frames - 1, target_frames, dtype=int)
        return data[indices]

    def _prepare_sketch_raw(self, sketch: np.ndarray) -> np.ndarray:
        """
        Raw MiDaS sketch: (T, H, W) uint8 0-255
        → (1, 256, T, 128, 128) float16

        No encoder — just resize, normalize, expand channels.
        Real near/far values preserved.
        """
        sketch = self._sample_frames(sketch, self.num_frames)  # (T, H, W)

        # Resize each frame
        W, H = self.resolution
        resized = []
        for t in range(sketch.shape[0]):
            frame = cv2.resize(sketch[t], (128, 128), interpolation=cv2.INTER_LINEAR)
            resized.append(frame)
        sketch = np.stack(resized)  # (T, 128, 128)

        # Normalize to [0, 1]
        sketch = sketch.astype(np.float32) / 255.0  # (T, 128, 128)

        # (T, 128, 128) → (1, 1, T, 128, 128) tensor
        tensor = torch.from_numpy(sketch).unsqueeze(0).unsqueeze(0)  # (1, 1, T, 128, 128)

        # Expand 1 channel → 256 channels (repeat same values)
        # Adapter expects (B, 256, T, H, W)
        tensor = tensor.expand(-1, 256, -1, -1, -1)  # (1, 256, T, 128, 128)

        return tensor.numpy().astype(np.float16)

    def process_single_file(self, file_info: dict) -> dict:
        result = {'success': False, 'size_mb': 0, 'errors': []}

        try:
            # Skip if already processed
            if file_info['output_path'].exists():
                result['success'] = True
                result['size_mb'] = file_info['output_path'].stat().st_size / 1e6
                return result

            data = np.load(file_info['input_path'], allow_pickle=True)

            if 'sketch' not in data:
                result['errors'].append("no sketch key in npz")
                return result

            sketch_encoded = self._prepare_sketch_raw(data['sketch'])

            np.savez_compressed(
                file_info['output_path'],
                sketch_encoded=sketch_encoded,
            )

            result['success'] = True
            result['size_mb'] = file_info['output_path'].stat().st_size / 1e6

        except Exception as e:
            result['errors'].append(f"general: {str(e)[:100]}")

        return result

    def process_all(self):
        print("=" * 70)
        print(f"sketch-Only Encoding (raw MiDaS, no encoder)")
        print(f"Files:  {len(self.npz_files)}")
        print(f"Frames: {self.num_frames}  |  Spatial: 128×128  |  Channels: 256")
        print("=" * 70 + "\n")

        success_count = 0
        total_size_mb = 0
        all_errors = []
        start_time = time.time()

        for file_info in tqdm(self.npz_files, desc="Encoding sketch", ncols=80):
            result = self.process_single_file(file_info)

            if result['success']:
                success_count += 1
                total_size_mb += result['size_mb']
            if result['errors']:
                all_errors.extend(result['errors'])

        elapsed = time.time() - start_time

        print(f"\n{'='*70}")
        print("Done!")
        print(f"  Success:      {success_count}/{len(self.npz_files)} files")
        print(f"  Total size:   {total_size_mb/1024:.2f} GB")
        print(f"  Time:         {elapsed/60:.1f} minutes")
        print(f"  Speed:        {len(self.npz_files)/elapsed:.1f} files/sec")

        if all_errors:
            print(f"\n  Errors: {len(all_errors)}")
            for err in all_errors[:10]:
                print(f"    - {err}")

        self._print_sample()

    def _print_sample(self):
        processed = list(self.output_dir.rglob('*_encoded.npz'))
        if not processed:
            print("No output files found.")
            return
        print(f"\nSample output ({processed[0].name}):")
        sample = np.load(processed[0])
        for key in sample.keys():
            d = sample[key]
            print(f"  {key:20s}: shape={str(d.shape):30s}  dtype={d.dtype}  "
                  f"min={d.min():.3f}  max={d.max():.3f}")
        print(f"\nTotal files: {len(processed)}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Encode sketch controls (raw MiDaS, no encoder)")
    parser.add_argument('--control_dir', type=str,
                        default='/mnt/d1/controllable-generation/control_signals',
                        help='Directory containing raw control NPZ files')
    parser.add_argument('--output_dir', type=str,
                        default='/mnt/d1/controllable-generation/encoded_controls',
                        help='Output directory for encoded sketch NPZ files')
    parser.add_argument('--num_frames', type=int, default=8)
    args = parser.parse_args()

    processor = SketchOnlyProcessor(
        control_base_dir=args.control_dir,
        output_dir=args.output_dir,
        num_frames=args.num_frames,
    )
    processor.process_all()


if __name__ == '__main__':
    main()