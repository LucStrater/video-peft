#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --job-name=recodec
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x_%A.out

# load environment
# load an FFmpeg build with the dnxhd encoder here

# Directories
INPUT_DIR="/path/to/datasets/NurVid/videos_original"
OUTPUT_DIR="/path/to/datasets/NurVid/videos_fps"

# Make sure output folder exists
mkdir -p "$OUTPUT_DIR"

# Loop over all mp4 files - just get filenames
cd "$INPUT_DIR"
for FILENAME in *.mp4; do
    # Skip if no files match
    [ -e "$FILENAME" ] || continue
    
    # Construct full paths
    INPUT="$INPUT_DIR/$FILENAME"
    BASENAME="${FILENAME%.mp4}"
    OUTPUT="$OUTPUT_DIR/$BASENAME.mov"
    
    # Skip if output already exists
    if [ -f "$OUTPUT" ]; then
        echo "Skipping $BASENAME (already exists)"
        continue
    fi
    
    # echo "Processing: $BASENAME"
    
    ffmpeg -hide_banner -loglevel warning -i "$INPUT" \
        -vf "scale='if(gt(dar,1),16*round((256*dar)/16),256)':'if(gt(dar,1),256,16*round((256/dar)/16))',setsar=1,format=yuv422p, fps=25" \
        -c:v dnxhd -profile:v dnxhr_hq -r 25 -vsync cfr -video_track_timescale 25000 -movflags +faststart -an \
        -y "$OUTPUT"
    
    if [ $? -eq 0 ]; then
        echo "Successfully converted: $BASENAME"
    else
        echo "ERROR converting: $BASENAME"
    fi
done

echo "All videos processed!"
