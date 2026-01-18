import json
import random
from pathlib import Path

def prepare_test_shots(
    annotation_path='../data/annotations/video_scripts_with_ref_filtered_changed_keys',
    num_test_shots=500,
    output_path='./test_data',
    max_json_files=10,
    seed=42
):
    """
    Select diverse test shots from your AnimeShooter dataset.
   
    """
    random.seed(seed)
    
 
    all_shots = []
    annotation_files = list(Path(annotation_path).glob('*.json'))
    
    
    annotation_files = annotation_files[:max_json_files]
    
    print(f"Processing {len(annotation_files)} JSON files (limited to {max_json_files})...")
    
    for i, ann_file in enumerate(annotation_files, 1):
        print(f"  Loading {i}/{len(annotation_files)}: {ann_file.name}")
        
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract video metadata
            video_id = data.get('video ID', 'unknown')
            video_url = data.get('url', '')
            fps = data.get('fps', 30.0)
            
            # Process segments
            segments = data.get('segments', [])
            shots_in_file = 0
            
            for segment in segments:
                story_script = segment.get('story script', {})
                shots = story_script.get('shots', [])
                
                # Add metadata to each shot
                for shot in shots:
                    # Calculate duration from start/end times
                    start_time = shot.get('start time', '00:00')
                    end_time = shot.get('end time', '00:00')
                    
                    # Parse time format "MM:SS"
                    start_parts = start_time.split(':')
                    end_parts = end_time.split(':')
                    
                    start_seconds = int(start_parts[0]) * 60 + int(start_parts[1])
                    end_seconds = int(end_parts[0]) * 60 + int(end_parts[1])
                    
                    duration = end_seconds - start_seconds
                    
                    # Add metadata to shot
                    shot_with_metadata = {
                        'video_id': video_id,
                        'video_url': video_url,
                        'video_file': f"{video_id}.mp4", 
                        'fps': fps,
                        'start_time': start_seconds,
                        'end_time': end_seconds,
                        'duration': duration,
                        'start_time_str': start_time,
                        'end_time_str': end_time,
                        'scene': shot.get('scene', 'unknown'),
                        'is_prologue_or_epilogue': shot.get('is_prologue_or_epilogue', False),
                        'main_characters': shot.get('main characters', []),
                        'narrative_caption': shot.get('visual annotation', {}).get('narrative caption', ''),
                        'descriptive_caption': shot.get('visual annotation', {}).get('descriptive caption', ''),
                    }
                    
                    all_shots.append(shot_with_metadata)
                    shots_in_file += 1
            
            print(f"    → Found {shots_in_file} shots in {len(segments)} segments")
            
        except Exception as e:
            print(f"    ✗ Error loading {ann_file.name}: {str(e)}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Found {len(all_shots)} total shots from {len(annotation_files)} files")
    print(f"{'='*60}")
  
    if len(all_shots) < num_test_shots:
        print(f"⚠️  Warning: Only {len(all_shots)} shots available, reducing test set size")
        num_test_shots = len(all_shots)
    
    
    short_shots = [s for s in all_shots if s['duration'] < 3]
    medium_shots = [s for s in all_shots if 3 <= s['duration'] < 6]
    long_shots = [s for s in all_shots if s['duration'] >= 6]
    
    print(f"\nShot duration distribution:")
    print(f"  - Short (<3s): {len(short_shots)}")
    print(f"  - Medium (3-6s): {len(medium_shots)}")
    print(f"  - Long (>6s): {len(long_shots)}")
    
   
    n_short = int(num_test_shots * 0.3)
    n_medium = int(num_test_shots * 0.5)
    n_long = num_test_shots - n_short - n_medium
    
    test_shots = (
        random.sample(short_shots, min(n_short, len(short_shots))) +
        random.sample(medium_shots, min(n_medium, len(medium_shots))) +
        random.sample(long_shots, min(n_long, len(long_shots)))
    )
    
   
    random.shuffle(test_shots)
    
  
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    
    output_data = {
        'num_shots': len(test_shots),
        'max_json_files_processed': len(annotation_files),
        'total_shots_available': len(all_shots),
        'duration_distribution': {
            'short': len([s for s in test_shots if s['duration'] < 3]),
            'medium': len([s for s in test_shots if 3 <= s['duration'] < 6]),
            'long': len([s for s in test_shots if s['duration'] >= 6])
        },
        'shots': test_shots
    }
    
    with open(output_path / 'test_shots.json', 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✓ Selected {len(test_shots)} test shots")
    print(f"  - Short (<3s): {output_data['duration_distribution']['short']}")
    print(f"  - Medium (3-6s): {output_data['duration_distribution']['medium']}")
    print(f"  - Long (>6s): {output_data['duration_distribution']['long']}")
    print(f"\n✓ Saved to: {output_path / 'test_shots.json'}")
    print(f"{'='*60}")
    
   
    print(f"\nSample shots:")
    for i, shot in enumerate(test_shots[:3], 1):
        print(f"\n  Shot {i}:")
        print(f"    Video: {shot['video_file']}")
        print(f"    Time: {shot['start_time_str']} - {shot['end_time_str']} ({shot['duration']}s)")
        print(f"    Caption: {shot['narrative_caption'][:60]}...")
    
    return test_shots


def verify_test_shots(test_shots_file='./test_data/test_shots.json'):
    """
    Verify the generated test shots file.
    """
    print(f"\n{'='*60}")
    print(f"VERIFYING TEST SHOTS")
    print(f"{'='*60}")
    
    with open(test_shots_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\nMetadata:")
    print(f"  Total shots: {data['num_shots']}")
    print(f"  Files processed: {data['max_json_files_processed']}")
    print(f"  Total available: {data['total_shots_available']}")
    
    print(f"\nDuration distribution:")
    print(f"  Short: {data['duration_distribution']['short']}")
    print(f"  Medium: {data['duration_distribution']['medium']}")
    print(f"  Long: {data['duration_distribution']['long']}")
    

    required_fields = [
        'video_id', 'video_file', 'start_time', 'end_time', 
        'duration', 'narrative_caption', 'descriptive_caption'
    ]
    
    shots = data['shots']
    print(f"\nField validation:")
    for field in required_fields:
        missing = sum(1 for s in shots if field not in s or s[field] is None)
        if missing > 0:
            print(f"  ⚠️  {field}: {missing} shots missing")
        else:
            print(f"  ✓ {field}: all present")
    
   
    durations = [s['duration'] for s in shots]
    print(f"\nDuration statistics:")
    print(f"  Min: {min(durations)}s")
    print(f"  Max: {max(durations)}s")
    print(f"  Average: {sum(durations)/len(durations):.2f}s")
    
   
    unique_videos = set(s['video_id'] for s in shots)
    print(f"\nUnique videos: {len(unique_videos)}")
    
    print(f"\n{'='*60}")
    print(f"✅ Verification complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
   
    test_shots = prepare_test_shots()
    
   
    verify_test_shots()