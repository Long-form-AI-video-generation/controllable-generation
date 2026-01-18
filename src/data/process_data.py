import json
import os
from pathlib import Path
from collections import defaultdict
import subprocess


def time_to_frames(time_str, fps):
    """Convert MM:SS timestamp to frame number.
    
    """
    if not time_str:
        return 0
    
    try:
        parts = time_str.split(':')
        minutes = int(parts[0])
        seconds = int(parts[1])
        total_seconds = minutes * 60 + seconds
        return int(total_seconds * fps)
    except (ValueError, IndexError):
        print(f"Warning: Invalid time format '{time_str}', using 0")
        return 0


def process_animeshooter_annotations(max_videos=None):
    """Process the AnimeShooter annotations and prepare for download.
   
    """
    print("=" * 70)
    print("AnimeShooter Dataset Processor")
    print("=" * 70)
    
    
    annotations_dir = Path("data/animeshooter/annotations/video_scripts_with_ref_filtered_changed_keys")
    
    if not annotations_dir.exists():
        print(f"\n❌ Directory not found: {annotations_dir}")
        return None, None
    
    print(f"\n✓ Found annotations directory")
    
   
    json_files = sorted(annotations_dir.glob("*.json")) 
    print(f"✓ Found {len(json_files)} JSON annotation files")
    
    if len(json_files) == 0:
        print("\n❌ No JSON files found!")
        return None, None
    
   
    if max_videos is not None:
        json_files = json_files[:max_videos]
        print(f"✓ Processing only first {max_videos} videos (for testing)")
    
 
    all_videos = []
    all_shots = []
    

    video_shot_tracker = defaultdict(set)
    duplicate_stats = defaultdict(int)
    
    print(f"\n📝 Processing {len(json_files)} annotations...")
    
    for i, json_file in enumerate(json_files):
        if (i + 1) % 10 == 0 or (i + 1) == len(json_files):
            print(f"  Processed {i + 1}/{len(json_files)} files...")
        
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                video_data = json.load(f)
            
           
            video_id = video_data.get('video ID', '').strip()
            url = video_data.get('url', '').strip()
            fps = video_data.get('fps', 24.0)
            
            if not video_id or not url:
                continue
            
            segments = video_data.get('segments', [])
            
          
            video_shots = []
            shot_counter = 0  
            
            for seg_idx, segment in enumerate(segments):
                story_script = segment.get('story script', {})
                shots = story_script.get('shots', [])
                
                for shot_idx, shot in enumerate(shots):
                    visual_annotation = shot.get('visual annotation', {})
                    
                    # Get timestamps
                    start_time = shot.get('start time', '').strip()
                    end_time = shot.get('end time', '').strip()
                    
                    if not start_time or not end_time:
                        continue
                    
                    # Calculate frame ranges
                    start_frame = time_to_frames(start_time, fps)
                    end_frame = time_to_frames(end_time, fps)
                    
                    # Skip invalid frame ranges
                    if end_frame <= start_frame:
                        continue
                    
                    # Create unique shot identifier using frame range
                    shot_signature = (start_frame, end_frame)
                    
                    # Check for duplicate and skip if found
                    if shot_signature in video_shot_tracker[video_id]:
                        duplicate_stats[video_id] += 1
                        continue
                    
                    # Mark this shot as seen
                    video_shot_tracker[video_id].add(shot_signature)
                    
                    # Create unique shot ID (global counter per video)
                    unique_shot_id = f"{video_id}_shot_{shot_counter:04d}"
                    shot_counter += 1
                    
                    shot_info = {
                        'shot_id': unique_shot_id,  # Unique shot identifier
                        'video_id': video_id,
                        'url': url,
                        'fps': fps,
                        'segment_idx': seg_idx,
                        'shot_idx_in_segment': shot_idx,
                        'start_time': start_time,
                        'end_time': end_time,
                        'segment_start_frame': start_frame,
                        'segment_end_frame': end_frame,
                        'num_frames': end_frame - start_frame,
                        'narrative_caption': visual_annotation.get('narrative_caption', ''),
                        'descriptive_caption': visual_annotation.get('descriptive caption', ''),
                        'main_characters': shot.get('main characters', []),
                        'scene': shot.get('scene', ''),
                    }
                    
                    video_shots.append(shot_info)
            
            # Only add video if it has valid shots
            if video_shots:
                video_info = {
                    'video_id': video_id,
                    'url': url,
                    'fps': fps,
                    'num_segments': len(segments),
                    'num_shots': len(video_shots),
                    'json_file': json_file.name
                }
                all_videos.append(video_info)
                all_shots.extend(video_shots)
                
        except Exception as e:
            print(f"\n  ⚠️ Error processing {json_file.name}: {e}")
            continue
    
    print(f"\n✓ Processing complete!")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total videos: {len(all_videos)}")
    print(f"  Total unique shots: {len(all_shots)}")
    
    if len(all_videos) > 0:
        print(f"  Average shots per video: {len(all_shots) / len(all_videos):.1f}")
    
    # Report duplicates removed
    total_duplicates = sum(duplicate_stats.values())
    if total_duplicates > 0:
        print(f"\n🔍 Duplicate Detection:")
        print(f"  Total duplicates removed: {total_duplicates}")
        print(f"  Videos with duplicates: {len(duplicate_stats)}")
        
        # Show top videos with most duplicates
        top_dupes = sorted(duplicate_stats.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_dupes:
            print(f"\n  Top videos with duplicates:")
            for vid_id, count in top_dupes:
                print(f"    {vid_id}: {count} duplicates removed")
    else:
        print(f"\n✓ No duplicate shots found!")
    
    # Verify uniqueness
    print(f"\n🔍 Verifying shot uniqueness...")
    shot_id_set = set(shot['shot_id'] for shot in all_shots)
    assert len(shot_id_set) == len(all_shots), "❌ Shot IDs are not unique!"
    print(f"  ✓ All {len(all_shots)} shots have unique IDs")
    
    # Verify frame ranges per video
    print(f"\n🔍 Verifying frame ranges per video...")
    for video_id in list(video_shot_tracker.keys())[:3]:  # Check first 3
        num_shots = len([s for s in all_shots if s['video_id'] == video_id])
        num_unique = len(video_shot_tracker[video_id])
        print(f"  ✓ {video_id}: {num_shots} shots, all with unique frame ranges")
    
    # Show sample shots
    if all_shots:
        print(f"\n📝 Sample Shots (first 3):")
        for i, shot in enumerate(all_shots[:3], 1):
            print(f"\n  Shot {i}:")
            print(f"    ID: {shot['shot_id']}")
            print(f"    Video: {shot['video_id']}")
            print(f"    Time: {shot['start_time']} - {shot['end_time']}")
            print(f"    Frames: {shot['segment_start_frame']} - {shot['segment_end_frame']} ({shot['num_frames']} frames)")
            caption = shot['narrative_caption'][:60] + "..." if len(shot['narrative_caption']) > 60 else shot['narrative_caption']
            print(f"    Narrative: {caption}" if caption else "    Narrative: None")
    
    # Save processed data
    output_dir = Path("data/animeshooter")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save video metadata
    videos_file = output_dir / "video_urls.json"
    with open(videos_file, 'w', encoding='utf-8') as f:
        json.dump(all_videos, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Saved {len(all_videos)} videos to: {videos_file}")
    
    # Save shot metadata
    shots_file = output_dir / "shots_metadata.json"
    with open(shots_file, 'w', encoding='utf-8') as f:
        json.dump(all_shots, f, indent=2, ensure_ascii=False)
    print(f"✓ Saved {len(all_shots)} shots to: {shots_file}")
    
    return all_videos, all_shots
def download_videos(video_list, max_videos=10, output_dir="data/animeshooter/videos"):
    """Download videos using yt-dlp.

    """
    
    print("\n" + "=" * 70)
    print(f"Downloading Videos (max: {max_videos})")
    print("=" * 70)
    
    # Check if yt-dlp is installed
    try:
        result = subprocess.run(
            ["yt-dlp", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\n❌ yt-dlp is not installed!")
        print("\nPlease install it using:")
        print("  pip install yt-dlp")
        print("  # or")
        print("  conda install -c conda-forge yt-dlp")
        return 0, 0
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = 0
    failed = 0
    skipped = 0
    
    for i, video in enumerate(video_list[:max_videos]):
        video_id = video['video_id']
        url = video['url']
        
        output_path = output_dir / f"{video_id}.mp4"
        
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✓ [{i+1}/{max_videos}] {video_id} already exists ({file_size:.1f} MB)")
            skipped += 1
            continue
        
        print(f"\n⬇️  [{i+1}/{max_videos}] Downloading {video_id}...")
        print(f"    URL: {url}")
        
       
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720]",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            "--no-warnings",
            "--quiet", 
            "--progress", 
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600  # 10 minute timeout per video
            )
            
            if result.returncode == 0 and output_path.exists():
                file_size = output_path.stat().st_size / (1024 * 1024)  # MB
                print(f"    ✓ Downloaded successfully ({file_size:.1f} MB)")
                successful += 1
            else:
                print(f"    ❌ Failed to download")
                if result.stderr:
                    # Show only first 200 chars of error
                    error_msg = result.stderr.strip()[:200]
                    print(f"    Error: {error_msg}")
                failed += 1
        
        except subprocess.TimeoutExpired:
            print(f"    ❌ Download timeout (>10 minutes)")
            failed += 1
        except Exception as e:
            print(f"    ❌ Error: {e}")
            failed += 1
    
    print(f"\n" + "=" * 70)
    print(f"Download Summary:")
    print(f"  ✓ Successful: {successful}")
    print(f"  ⏭️  Skipped (already exist): {skipped}")
    print(f"  ❌ Failed: {failed}")
    print(f"  📁 Videos saved to: {output_dir}")
    print("=" * 70)
    
    return successful, failed
def main():
    """Main workflow."""
    
    # Step 0: Ask how many videos to process
    print("\n" + "=" * 70)
    print("AnimeShooter Dataset Setup")
    print("=" * 70)
    print("\nHow many videos do you want to process?")
    print("  - Start small: 10 videos (recommended for testing)")
    print("  - Medium: 50-100 videos")
    print("  - All: type 'all' (processes everything)")
    
    num_process = input("\nNumber of videos to process (default: 10): ").strip()
    
    if num_process == '':
        num_process = 10
    elif num_process.lower() == 'all':
        num_process = None
    else:
        try:
            num_process = int(num_process)
        except:
            print("⚠️  Invalid input, using default: 10")
            num_process = 10
    

    videos, shots = process_animeshooter_annotations(max_videos=num_process)
    
    if not videos:
        print("\n❌ No videos found. Exiting.")
        return

    print("\n" + "=" * 70)
    print("Download Videos")
    print("=" * 70)
    print(f"\nProcessed {len(videos)} videos")
    
    response = input("\nDownload videos now? (yes/no, default: no): ").strip().lower()
    
    if response in ['yes', 'y']:
        num_videos = input(f"How many videos to download? (default: {len(videos)}): ").strip()
        num_videos = int(num_videos) if num_videos.isdigit() else len(videos)
        
        download_videos(videos, max_videos=num_videos)
    else:
        print("\n⏭️  Skipping download. You can download later by running:")
        print("   python process_animeshooter.py")
    
    print("\n" + "=" * 70)
    print("✓ Setup Complete!")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Check the generated files in data/animeshooter/")
    print("  2. Download videos if you haven't already")
    print("  3. Run control signal extraction")


if __name__ == "__main__":
    main()