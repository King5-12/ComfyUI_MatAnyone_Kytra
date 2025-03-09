"""
Example script showing how to use MatAnyone programmatically.
"""
import os
import torch
from inference_matanyone import process_video

# Enable CUDA if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define paths
input_video = "inputs/video/my_video.mp4"  # Replace with your video path
mask_image = "inputs/mask/my_mask.png"     # Replace with your mask path
output_dir = "results"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process the video
fgr_path, pha_path = process_video(
    input_path=input_video,
    mask_path=mask_image,
    output_path=output_dir,
    n_warmup=10,           # Number of warmup iterations
    r_erode=10,            # Erosion kernel size
    r_dilate=10,           # Dilation kernel size
    max_size=1280,         # Limit the maximum size (optional)
    save_image=False       # Whether to save individual frames
)

print(f"Processing complete!")
print(f"Foreground video saved to: {fgr_path}")
print(f"Alpha matte saved to: {pha_path}")

# Example of how to use the outputs
print("\nNext steps:")
print("1. Use the foreground video (fgr) with a green background for chroma keying")
print("2. Use the alpha matte (pha) as a transparency mask in video editing software")
print("3. Combine them to composite your subject onto a new background") 