"""
Example workflow for using MatAnyone in ComfyUI

This is a Python representation of a ComfyUI workflow.
You can use this as a reference for how to connect nodes.
"""

import torch
import numpy as np
from PIL import Image
import os

# This is a representation of how you would use the node in ComfyUI
# In the actual ComfyUI interface, you would connect nodes visually

def example_workflow():
    # Step 1: Load video frames
    # In ComfyUI, you would use a video loader node
    # Here we're just creating dummy frames for illustration
    video_frames = torch.rand(10, 3, 512, 512)  # 10 frames, RGB, 512x512
    
    # Step 2: Create a mask for the first frame
    # In ComfyUI, you would use a mask drawing tool or load a mask
    mask = torch.zeros(512, 512)
    # Draw a circle in the center as an example foreground
    center_x, center_y = 256, 256
    radius = 100
    for y in range(512):
        for x in range(512):
            if (x - center_x)**2 + (y - center_y)**2 < radius**2:
                mask[y, x] = 1.0
    
    # Step 3: Process with MatAnyone
    # In ComfyUI, this would be the MatAnyone node
    from __init__ import MatAnyoneNode
    matanyone_node = MatAnyoneNode()
    
    # Make sure the model is downloaded
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "matanyone.pth")
    
    if not os.path.exists(model_path):
        print(f"Please download the model from https://github.com/pq-yang/MatAnyone/releases/download/v1.0.0/matanyone.pth")
        print(f"and place it in {model_dir}")
        return
    
    # Process the video
    foreground_frames, alpha_frames = matanyone_node.process_video(
        video_frames=video_frames,
        mask=mask,
        warmup_frames=10,
        erode_kernel=10,
        dilate_kernel=10,
        max_size=-1
    )
    
    # Step 4: Use the results
    # In ComfyUI, you would connect these to other nodes
    print(f"Foreground frames shape: {foreground_frames.shape}")
    print(f"Alpha frames shape: {alpha_frames.shape}")
    
    # Example: Save the first frame as an image
    fg_frame = foreground_frames[0].permute(1, 2, 0).cpu().numpy()
    fg_frame = (fg_frame * 255).astype(np.uint8)
    fg_image = Image.fromarray(fg_frame)
    fg_image.save("example_foreground.png")
    
    alpha_frame = alpha_frames[0][0].cpu().numpy()
    alpha_frame = (alpha_frame * 255).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_frame)
    alpha_image.save("example_alpha.png")
    
    print("Example images saved!")

if __name__ == "__main__":
    example_workflow() 