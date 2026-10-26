import json
import os
import subprocess

def get_video_info(video_path):
    """Get video duration and frame rate using ffprobe"""
    # Get duration
    duration_command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    # Get frame rate
    fps_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    # Get codec to verify it's DNxHR
    codec_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]
    
    try:
        duration_result = subprocess.run(duration_command, capture_output=True, text=True, check=True)
        duration = float(duration_result.stdout.strip())
        
        fps_result = subprocess.run(fps_command, capture_output=True, text=True, check=True)
        fps_str = fps_result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            fps = num / den
        else:
            fps = float(fps_str)
        
        codec_result = subprocess.run(codec_command, capture_output=True, text=True, check=True)
        codec = codec_result.stdout.strip()
        
        return duration, fps, codec
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"Error getting video info for {video_path}: {e}")
        return None, None, None

with open("/path/to/datasets/NurVid/NurViD_annotations.json", "r") as f:
    data = json.load(f)

video_names = data.keys()
skipped_count = 0
adjusted_count = 0

for video_name in video_names:
    video_path = data[video_name]["url"]
    video_name = video_path.split("=")[-1]
    video_dir = "/path/to/datasets/NurVid/videos_fps"
    video_path = os.path.join(video_dir, video_name+'.mov')
    
    if not os.path.isfile(video_path):
        print(f"WARNING: Video file not found: {video_path}")
        continue
    
    video_duration, fps, codec = get_video_info(video_path)
    if video_duration is None:
        print(f"WARNING: Could not determine duration for {video_path}, skipping all annotations")
        continue
    
    # Verify codec
    if codec and 'dnxhd' not in codec.lower() and 'dnxhr' not in codec.lower():
        print(f"WARNING: {video_path} is {codec}, not DNxHR/DNxHD - stream copy may be less accurate")
    
    operation_id = data[video_name]["operationID"]
    annotations = data[video_name]["annotations"]
    
    for i, annotation in enumerate(annotations):
        action_id = annotation["actionID"]
        start_time = annotation["segment"][0]
        end_time = annotation["segment"][1]
        
        if start_time > video_duration and end_time > video_duration:
            print(f"SKIPPING {video_name}_{i+1}: Both start ({start_time:.2f}s) and end ({end_time:.2f}s) exceed video duration ({video_duration:.2f}s)")
            skipped_count += 1
            continue
        
        if end_time > video_duration:
            print(f"ADJUSTING {video_name}_{i+1}: End time {end_time:.2f}s exceeds video duration ({video_duration:.2f}s), setting to video duration")
            end_time = video_duration
            adjusted_count += 1
        
        duration = end_time - start_time
        
        new_video_dir = "/path/to/datasets/NurVid/videos"
        new_video_name = os.path.join(new_video_dir, video_name + "_" + str(i + 1) + ".mov")
        
        if os.path.isfile(new_video_name):
            print(f"Skipping {video_name}_{i+1} (already exists)")
            continue
        
        # Optimized for DNxHR - stream copy with accurate timing
        command = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-ss", str(start_time),      # Seek to start time
            "-t", str(duration),          # Duration of clip
            "-i", video_path,
            "-c", "copy",                 # Stream copy - lossless and fast
            "-avoid_negative_ts", "make_zero",  # Fix any timestamp issues
            "-map_metadata", "0",         # Copy metadata
            "-map", "0",                  # Copy all streams
            "-movflags", "+faststart",    # Enable fast start for playback
            "-fflags", "+genpts",         # Generate presentation timestamps
            "-y",
            new_video_name
        ]
        
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Successfully processed: {video_name}_{i+1}")
        else:
            print(f"ERROR processing: {video_name}_{i+1}")
            print(f"Error: {result.stderr}")

print(f"\nAll videos processed!")
print(f"Total annotations skipped (both times exceed duration): {skipped_count}")
print(f"Total annotations adjusted (end time reset to video duration): {adjusted_count}")