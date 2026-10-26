import json

with open('/path/to/datasets/NurVid/NurViD_annotations.json', 'r') as f:
    data = json.load(f)

with open('/path/to/datasets/NurVid/train_old.json', 'r') as f:
    filter_data = json.load(f)

valid_vid_names = {item['vid_name'] for item in filter_data}

# Transform the data
transformed_data = []
skipped_segments = []
filtered_out = []

for video_id, video_info in data.items():
    operation_id = video_info['operationID']
    label = operation_id - 1  # label is operationID - 1
    duration = video_info['duration']
    
    # Process each annotation segment
    seg_id = 1
    for annotation in video_info['annotations']:
        start_time = annotation['segment'][0]
        end_time = annotation['segment'][1]
        
        # Check if segment times are valid (within duration)
        if start_time <= duration and int(end_time) <= duration:
            vid_name = f"{video_id}_{seg_id}"
            
            # Only add entry if it exists in the filter file
            if vid_name in valid_vid_names:
                entry = {
                    "vid_name": video_id,
                    "label": label,
                    "start_time": start_time,
                    "end_time": end_time
                }
                transformed_data.append(entry)
            else:
                filtered_out.append(vid_name)
            
            seg_id += 1
        else:
            skipped_segments.append({
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "duration": duration
            })

# Write the transformed data to output file
# Update this path to your desired output location
with open('/path/to/datasets/NurVid/train.json', 'w') as f:
    json.dump(transformed_data, f, indent=2)

print(f"Transformation complete! Created {len(transformed_data)} entries.")
if skipped_segments:
    print(f"\nSkipped {len(skipped_segments)} segments that exceeded video duration:")
    for skipped in skipped_segments:
        print(f"  Video: {skipped['video_id']}, Start: {skipped['start_time']}, End: {skipped['end_time']}, Duration: {skipped['duration']}")
