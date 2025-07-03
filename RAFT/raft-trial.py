import sys
# The RAFT model is in the `core` directory, so we add it to the path
sys.path.append('core')

import argparse
import os
import cv2
import numpy as np
import torch

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile, target_size=None):
    """Loads an image, resizes if needed, converts to RGB, and moves to device."""
    img = cv2.imread(imfile)
    if img is None:
        print(f"Error: Could not read image file {imfile}")
        return None

    # --- START OF MODIFICATION ---
    if target_size:
        # Resize while maintaining aspect ratio
        h, w, _ = img.shape
        aspect_ratio = w / h
        
        # New width will be target_size[0], calculate new height
        new_w = target_size[0]
        new_h = int(new_w / aspect_ratio)

        # Ensure the new height is divisible by 8 (a requirement for RAFT)
        new_h = (new_h // 8) * 8
        
        if new_h > 0: # Check to avoid zero height
             img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    # --- END OF MODIFICATION ---

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def main(args):
    # 1. Setup Model
    print("Loading RAFT model...")
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model loaded.")

    with torch.no_grad():
        # Get all image pairs in the specified folder
        images = os.listdir(args.path)
        images = sorted(images)
        
        # We assume images are named sequentially (e.g., 001.png, 002.png)
        for i in range(len(images) - 1):
            image1_path = os.path.join(args.path, images[i])
            image2_path = os.path.join(args.path, images[i+1])

            print(f"Processing pair: {images[i]} and {images[i+1]}")
            
            # 2. Load and Preprocess Images
            target_width = 512 
            image1 = load_image(image1_path, target_size=(target_width, None))
            image2 = load_image(image2_path, target_size=(target_width, None))

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            # 3. Run RAFT Inference
            # The model returns a list of flow predictions. We use the final one.
            _, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # 4. Post-process and Visualize
            # Move flow to CPU and convert to numpy
            flow_up = padder.unpad(flow_up)
            flo = flow_up[0].permute(1, 2, 0).cpu().numpy()

            # Convert flow to a color image
            flo_img = flow_viz.flow_to_image(flo)
            
            # You can also use the original image's BGR format if you want
            # We convert our visualized flow map to BGR to save with cv2
            flo_img_bgr = cv2.cvtColor(flo_img, cv2.COLOR_RGB2BGR)

            # Save the visualization
            output_filename = f"flow_{os.path.splitext(images[i])[0]}_to_{os.path.splitext(images[i+1])[0]}.png"
            output_path = os.path.join(args.output, output_filename)
            cv2.imwrite(output_path, flo_img_bgr)
            
            print(f"Saved flow visualization to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/raft-sintel.pth', help="restore checkpoint")
    parser.add_argument('--path', required=True, help="path to a folder of images")
    parser.add_argument('--output', default='demo-output', help="path to save flow visualizations")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    main(args)