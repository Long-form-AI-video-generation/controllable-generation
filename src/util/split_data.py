import json
from pathlib import Path
from collections import defaultdict


def analyze_dataset_split(annotations_path: str):
    
    
    with open(annotations_path) as f:
        all_annotations = json.load(f)
    
  
    videos_with_shots = defaultdict(list)
    for ann in all_annotations:
        video_id = ann['video_id']
        videos_with_shots[video_id].append(ann)
   
    all_video_ids = sorted(videos_with_shots.keys())
  
    num_videos = len(all_video_ids)
    train_end = int(num_videos * 0.7)
    val_end = int(num_videos * 0.9)
    
    train_videos = all_video_ids[:train_end]
    val_videos = all_video_ids[train_end:val_end]
    test_videos = all_video_ids[val_end:]
    
    train_shots = sum(len(videos_with_shots[v]) for v in train_videos)
    val_shots = sum(len(videos_with_shots[v]) for v in val_videos)
    test_shots = sum(len(videos_with_shots[v]) for v in test_videos)
    total_shots = len(all_annotations)
    
    print("="*70)
    print("Dataset Split Analysis")
    print("="*70)
    print(f"\nTotal Videos: {num_videos}")
    print(f"Total Shots: {total_shots}")
    print(f"\nSplit Ratios: 70% train / 20% val / 10% test")
    print("\nVideos:")
    print(f"  Train: {len(train_videos)} videos ({len(train_videos)/num_videos*100:.1f}%)")
    print(f"  Val:   {len(val_videos)} videos ({len(val_videos)/num_videos*100:.1f}%)")
    print(f"  Test:  {len(test_videos)} videos ({len(test_videos)/num_videos*100:.1f}%)")
    print("\nShots:")
    print(f"  Train: {train_shots} shots ({train_shots/total_shots*100:.1f}%)")
    print(f"  Val:   {val_shots} shots ({val_shots/total_shots*100:.1f}%)")
    print(f"  Test:  {test_shots} shots ({test_shots/total_shots*100:.1f}%)")
    
    print(f"\nFirst 5 train videos: {train_videos[:5]}")
    print(f"First 5 val videos: {val_videos[:5]}")
    print(f"First 5 test videos: {test_videos[:5]}")
    print("="*70)
    
    return {
        'train': train_videos,
        'val': val_videos,
        'test': test_videos
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotations', type=str, 
                       default='data/shots_metadata.json')
    args = parser.parse_args()
    
    analyze_dataset_split(args.annotations)