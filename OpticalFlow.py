import sys
import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

# Add RAFT module to path
sys.path.append('RAFT')

from raft import RAFT
from utils.utils import InputPadder
from utils import flow_viz

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def viz_flow(flow):
    flow_img = flow_viz.flow_to_image(flow[0].permute(1,2,0).cpu().numpy())
    plt.imshow(flow_img)
    plt.title("Optical Flow")
    plt.axis('off')
    plt.show()

def main(img1_path, img2_path):
    # Load model
    model = torch.nn.DataParallel(RAFT())
    model.load_state_dict(torch.load('raft-sintel.pth'))
    model = model.module
    model.to(DEVICE)
    model.eval()

    # Load and pad images
    image1 = load_image(img1_path)
    image2 = load_image(img2_path)
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    # Compute flow
    with torch.no_grad():
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)

    # Visualize
    viz_flow(flow_up)

# Usage example
main('frame1.png', 'frame2.png')
