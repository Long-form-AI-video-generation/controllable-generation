import json
from collections import defaultdict

with open('../../data/shots_metadata.json', 'r') as f:
    data = json.load(f)


videos = defaultdict(list)
for shot in data:
    videos[shot['video_id']].append(shot)


print("=" * 60)
for video_id, shots in videos.items():
    total_frames = sum(shot['num_frames'] for shot in shots)
    num_shots = len(shots)
    fps = shots[0]['fps']
    
    avg_frames_per_shot = total_frames / num_shots
    avg_seconds_per_shot = avg_frames_per_shot / fps
    total_video_length = total_frames / fps
    
    print(f"Video ID: {video_id}")
    print(f"  Total shots: {num_shots}")
    print(f"  Average shot length: {avg_seconds_per_shot:.2f} seconds")
    print(f"  Total video length: {total_video_length:.2f} seconds ({total_video_length/60:.2f} minutes)")
    print("-" * 60)

# Overall statistics
total_frames_all = sum(shot['num_frames'] for shot in data)
total_shots_all = len(data)
avg_frames_all = total_frames_all / total_shots_all
avg_seconds_all = avg_frames_all / data[0]['fps']

print("\nOVERALL STATISTICS:")
print(f"Total videos: {len(videos)}")
print(f"Total shots: {total_shots_all}")
print(f"Average shot length across all videos: {avg_seconds_all:.2f} seconds")
print("=" * 60)