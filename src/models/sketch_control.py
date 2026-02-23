import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm
import warnings
from collections import defaultdict
import gc
from PIL import Image

warnings.filterwarnings('ignore')



def extract_sketch_anime_lineart(frame_bgr):
    
    from controlnet_aux import LineartAnimeDetector
    detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
    pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    result = detector(pil)
    return np.array(result.convert('L'))  


def extract_sketch_xdog(frame_bgr, sigma=0.5, k=4.5, p=19, epsilon=-0.1, phi=10e9):
   
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)

    dog = g1 - p * g2

    
    sketch = np.where(
        dog >= epsilon,
        1.0,
        1.0 + np.tanh(phi * dog)
    )

    sketch = np.clip(sketch, 0, 1)
    sketch = (sketch * 255).astype(np.uint8)

    return cv2.bitwise_not(sketch)


def extract_sketch_canny_anime(frame_bgr):
   
    smooth = cv2.bilateralFilter(frame_bgr, d=9, sigmaColor=75, sigmaSpace=75)
    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    return edges  


class SketchExtractor:
    
    def __init__(self, method='auto', device='cuda'):
        self.device = device
        self.method = method
        self.detector = None

        print(f"  Initializing sketch extractor (method={method})…")

        if method in ('auto', 'lineart_anime'):
            self.detector = self._try_load_lineart_anime()

        if self.detector is None:
            if method == 'lineart_anime':
                print("    controlnet-aux not available, falling back to XDoG")
            self.method = 'xdog'
            print(f"    Using XDoG sketch extraction (no dependencies needed)")

    def _try_load_lineart_anime(self):
        try:
            from controlnet_aux import LineartAnimeDetector
            detector = LineartAnimeDetector.from_pretrained("lllyasviel/Annotators")
            print("   Using ControlNet anime lineart detector")
            self.method = 'lineart_anime'
            return detector
        except Exception as e:
            print(f"   controlnet-aux not available ({e})")
            return None

    def extract(self, frame_bgr, target_size=(360, 640)):
        
        h, w = target_size

        if self.method == 'lineart_anime' and self.detector is not None:
            pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            result = self.detector(pil, detect_resolution=512, image_resolution=max(h, w))
            sketch = np.array(result.convert('L'))
        elif self.method == 'xdog':
            sketch = extract_sketch_xdog(frame_bgr)
        else:
            sketch = extract_sketch_canny_anime(frame_bgr)

       
        sketch = cv2.resize(sketch, (w, h), interpolation=cv2.INTER_LINEAR)
        return sketch 


def process_shot_sketch_only(
    video_path,
    shot,
    output_dir,
    extractor: SketchExtractor,
    sample_rate=4,
    target_size=(360, 640),
):
    video_id    = shot['video_id']
    shot_idx    = shot['shot_id']
    start_frame = shot['segment_start_frame']



    
    end_frame   = shot['segment_end_frame']
    num_frames  = end_frame - start_frame

    if num_frames < sample_rate:
        return False

    out_path = Path(output_dir) / video_id
    out_path.mkdir(parents=True, exist_ok=True)
    output_file = out_path / f"shot_{shot_idx}_controls_encoded.npz"

    if output_file.exists():
        return True

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    try:
        sketches = []
        frames_to_process = list(range(0, num_frames, sample_rate))
        pbar = tqdm(frames_to_process, desc=f"  Shot {shot_idx}", leave=False, unit="frame")

        for frame_offset in frames_to_process:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame + frame_offset)
            ret, frame = cap.read()
            if not ret:
                break

            frame_resized = cv2.resize(frame, target_size[::-1])  
            sketch = extractor.extract(frame_resized, target_size)
            sketches.append(sketch)

        pbar.close()

        if not sketches:
            return False

        np.savez_compressed(
            output_file,
            sketch=np.stack(sketches),  
            metadata={
                'video_id':    video_id,
                'shot_id':     shot_idx,
                'sample_rate': sample_rate,
                'target_size': target_size,
                'sketch_method': extractor.method,
            }
        )
        return True

    except Exception as e:
        print(f"\n  Shot {shot_idx}: {e}")
        import traceback; traceback.print_exc()
        return False
    finally:
        cap.release()
        gc.collect()



def process_dataset(
    video_dir,
    shot_dir,
    output_dir,
    sample_rate=4,
    target_size=(360, 640),
    sketch_method='auto',
    device='cuda',
):
    print(f"\n{'='*70}")
    print(f"Sketch Extraction  (method={sketch_method})")
    print(f"{'='*70}\n")

    with open(shot_dir) as f:
        all_shots = json.load(f)

    shots_by_video = defaultdict(list)
    for shot in all_shots:
        shots_by_video[shot['video_id']].append(shot)

    print(f"Loaded {len(all_shots)} shots from {len(shots_by_video)} videos\n")

    extractor = SketchExtractor(method=sketch_method, device=device)

    total_ok = total_fail = 0
    pbar = tqdm(shots_by_video.items(), desc="Videos", unit="video")

    for video_id, shots in pbar:
        video_path = Path(video_dir) / f"{video_id}.mp4"
        if not video_path.exists():
            total_fail += len(shots)
            continue

        for shot in shots:
            ok = process_shot_sketch_only(
                video_path, shot, output_dir,
                extractor, sample_rate, target_size,
            )
            total_ok   += ok
            total_fail += not ok

        pbar.set_postfix(Processed=total_ok, Failed=total_fail)

    print(f"\n{'='*70}")
    print(f"Complete!  Processed: {total_ok}  |  Failed: {total_fail}")
    print(f"{'='*70}\n")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract sketch/lineart control signals")
    parser.add_argument('--shots',       default="/mnt/d1/controllable-generation/shots_metadata.json")
    parser.add_argument('--videos',      default="/mnt/d1/controllable-generation/videos")
    parser.add_argument('--output',      default="/mnt/d1/controllable-generation/encoded_controls")
    parser.add_argument('--sample_rate', type=int, default=4)
    parser.add_argument('--method',      default='auto',
                        choices=['auto', 'lineart_anime', 'xdog', 'canny'],
                        help='auto tries lineart_anime first, falls back to xdog')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {DEVICE}\n")

    process_dataset(
        video_dir=args.videos,
        shot_dir=args.shots,
        output_dir=args.output,
        sample_rate=args.sample_rate,
        sketch_method=args.method,
        device=DEVICE,
    )


if __name__ == "__main__":
    main()