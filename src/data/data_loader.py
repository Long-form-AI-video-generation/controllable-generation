
import json
from pathlib import Path
from typing import Dict, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_animeshooter_simple(save_dir: str = "./data"):
    """
    Load AnimeShooter dataset using HuggingFace datasets library.
    
    """
    
    print("=" * 70)
    print("Loading AnimeShooter Dataset (Simple Method)")
    print("=" * 70)
    
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("\nInstalling datasets library...")
        import subprocess
        subprocess.run([
            "pip", "install", "datasets", "--break-system-packages"
        ], check=True)
        from datasets import load_dataset
    
    # Create save directory
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("\n[1/3] Loading dataset from HuggingFace...")
    print("(This will download ~6GB, please wait...)")
    
    try:
        dataset = load_dataset(
            "qiulu66/AnimeShooter",
            cache_dir=str(save_dir / "cache")
        )
        
        print(f"\nDataset loaded successfully!")
        print(f"\nDataset structure:")
        print(dataset)
        
        # Access the data
        if hasattr(dataset, 'data'):
            print(f"\n Dataset has 'data' attribute")
            data = dataset.data
        elif isinstance(dataset, dict):
            print(f"\nDataset is a dictionary with keys: {list(dataset.keys())}")
            if 'train' in dataset:
                data = dataset['train']
            else:
                data = dataset[list(dataset.keys())[0]]
        else:
            data = dataset
        
        print(f"\n[2/3] Extracting metadata...")
        
        # Get number of examples
        num_examples = len(data)
        print(f"Total examples: {num_examples}")
        
        # Show first example
        if num_examples > 0:
            print(f"\n[3/3] Sample data structure:")
            print("-" * 70)
            
            sample = data[0]
            print(f"Sample keys: {list(sample.keys())}")
            
            # Show sample data
            for key, value in sample.items():
                if isinstance(value, (str, int, float)):
                    print(f"{key}: {value}")
                elif isinstance(value, list):
                    print(f"{key}: [list with {len(value)} items]")
                elif isinstance(value, dict):
                    print(f"{key}: [dict with keys: {list(value.keys())}]")
                else:
                    print(f"{key}: {type(value)}")
            
            # Save sample
            sample_file = save_dir / "sample_data.json"
            with open(sample_file, 'w', encoding='utf-8') as f:
                
                sample_dict = {}
                for k, v in sample.items():
                    if isinstance(v, (str, int, float, list, dict)):
                        sample_dict[k] = v
                    else:
                        sample_dict[k] = str(v)
                
                json.dump(sample_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\n✅ Saved sample to: {sample_file}")
        
        # Process and save metadata
        print(f"\n{'='*70}")
        print("Extracting video and shot information...")
        print(f"{'='*70}")
        
        all_videos = []
        all_shots = []
        
        for idx, example in enumerate(data):
            if idx >= 100:  # Limit for initial testing
                print(f"\n(Stopping at 100 examples for testing)")
                break
            
            # Extract video info
            video_info = {
                'idx': idx,
                'video_id': example.get('video_id', example.get('id', f'video_{idx}')),
                'url': example.get('url', ''),
            }
            
           
            for key in example.keys():
                if 'segment' in key.lower() or 'shot' in key.lower() or 'annotation' in key.lower():
                    video_info[key] = example[key]
            
            all_videos.append(video_info)
            
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} examples...")
        
        # Save metadata
        videos_file = save_dir / "videos_metadata.json"
        with open(videos_file, 'w', encoding='utf-8') as f:
            json.dump(all_videos, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Saved metadata for {len(all_videos)} videos to: {videos_file}")
        
        
        
        return dataset, all_videos
        
    except Exception as e:
        print(f"\n Error loading dataset: {e}")
       
        raise


def download_videos_from_metadata(
    metadata_file: str,
    output_dir: str = "./data/videos",
    max_videos: int = 10
):
    """
    Download videos from the extracted metadata.
    """
    import subprocess
    
    print("=" * 70)
    print(f"Downloading Videos (max: {max_videos})")
    print("=" * 70)
    
    # Load metadata
    with open(metadata_file, 'r') as f:
        videos = json.load(f)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    
    for i, video in enumerate(videos[:max_videos]):
        video_id = video.get('video_id', f'video_{i}')
        url = video.get('url', '')
        
        if not url:
            print(f"No URL for {video_id}, skipping")
            failed += 1
            continue
        
        output_path = output_dir / f"{video_id}.mp4"
        
        if output_path.exists():
            print(f"✅ [{i+1}/{max_videos}] {video_id} already exists")
            successful += 1
            continue
        
        print(f"{i+1}/{max_videos}] Downloading {video_id}...")
        
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            url
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=600)
            
            if result.returncode == 0 and output_path.exists():
                print(f"✅ Downloaded {video_id}")
                successful += 1
            else:
                print(f"Failed to download {video_id}")
                failed += 1
        except Exception as e:
            print(f"Error downloading {video_id}: {e}")
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"Download complete: {successful} successful, {failed} failed")
    print(f"{'='*70}")


if __name__ == "__main__":
    import sys
    
    # Load dataset
    dataset, videos = load_animeshooter_simple()
    
    # Ask about downloading
    print("\n" + "=" * 70)
    response = input("Download videos now? (yes/no): ").strip().lower()
    
    if response in ['yes', 'y']:
        max_vids = int(input("How many videos to download? (recommended: 10): ").strip() or "10")
        download_videos_from_metadata(
            "./data/animeshooter/videos_metadata.json",
            max_videos=max_vids
        )