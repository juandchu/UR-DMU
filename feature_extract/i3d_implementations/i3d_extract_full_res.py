import os
import math
import gc
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from i3dpt import I3D

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

CHUNK_SIZE = 16  
FREQUENCY = 16  
MIN_FRAMES = 16  


########################################
# 1. PYTORCH DATASET WITH NATIVE GRID CROPPING
########################################
class NativeGridVideoDataset(Dataset):
    def __init__(self, frames_dir, rgb_files, frame_indices, crop_size=224, stride=112):
        self.frames_dir = frames_dir
        self.rgb_files = rgb_files
        self.frame_indices = frame_indices
        self.crop_size = crop_size
        self.stride = stride

        # Removed the Resize transform completely.
        # This forces the model to load the exact original pixels, preserving small objects.
        self.transform = T.Compose([
            T.ToTensor(),  
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  
        ])

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        indices = self.frame_indices[idx]
        snippet = []

        # Load the 16 frames in their native resolution
        for frame_idx in indices:
            img_path = os.path.join(self.frames_dir, self.rgb_files[frame_idx])
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img)  
            snippet.append(img_tensor)

        # Stack frames into shape: (Channels, Time, Height, Width)
        snippet_tensor = torch.stack(snippet, dim=1) 
        _, _, H, W = snippet_tensor.shape

        # Dynamically calculate grid starting points based on the native resolution.
        # The stride determines how much the boxes overlap.
        y_coords = list(range(0, H - self.crop_size + 1, self.stride))
        x_coords = list(range(0, W - self.crop_size + 1, self.stride))

        # Ensure we capture the absolute bottom and right edges if the math does not divide evenly.
        if y_coords[-1] + self.crop_size < H:
            y_coords.append(H - self.crop_size)
        if x_coords[-1] + self.crop_size < W:
            x_coords.append(W - self.crop_size)

        # Extract every square in the grid
        crops = []
        for y in y_coords:
            for x in x_coords:
                crop = snippet_tensor[:, :, y:y+self.crop_size, x:x+self.crop_size]
                crops.append(crop)

        # Stack all crops into a new batch dimension.
        # Shape returned is dynamic: (num_grid_crops, 3, 16, 224, 224)
        return torch.stack(crops, dim=0)


########################################
# 2. MODEL INITIALIZATION
########################################
i3d_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_model(load_model_path):
    global i3d_model
    i3d_model = I3D(400, modality="rgb", dropout_prob=0, name="inception").to(device)
    i3d_model.eval()

    checkpoint = torch.load(load_model_path, map_location=device, weights_only=True)
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    state = {k: v for k, v in state.items() if not k.startswith("conv3d_0c_1x1.")}

    i3d_model.load_state_dict(state, strict=False)
    i3d_model.eval()


########################################
# 3. MAIN EXTRACTION FUNCTION
########################################
def run(video_dir, input_root, output_root, batch_size, num_workers):
    rel_path = os.path.relpath(video_dir, input_root)
    save_path = os.path.join(output_root, rel_path + "_i3d.npy")
    video_name = os.path.basename(video_dir)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    if os.path.exists(save_path):
        print(f"{save_path} already exists.")
        return

    rgb_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith("jpg")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    frame_cnt = len(rgb_files)
    if frame_cnt <= MIN_FRAMES:
        print(f"{video_name}: only {frame_cnt} frames. Skipping...")
        return

    T_chunks = math.ceil(frame_cnt / CHUNK_SIZE)
    pad = (T_chunks * FREQUENCY) - frame_cnt
    if pad > 0:
        rgb_files += [rgb_files[-1]] * pad

    frame_indices = np.array(
        [list(range(i * FREQUENCY, i * FREQUENCY + CHUNK_SIZE)) for i in range(T_chunks)]
    )

    # Use the new Native Grid Dataset
    dataset = NativeGridVideoDataset(video_dir, rgb_files, frame_indices)
    
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  
    )

    all_features = []
    global i3d_model

    for batch_data in tqdm(dataloader, desc=video_name):
        
        # num_crops is no longer exactly 10. It depends on the video resolution.
        B, num_crops, C, T_dim, H, W = batch_data.shape
        
        batch_data = batch_data.view(B * num_crops, C, T_dim, H, W).to(device)

        with torch.no_grad():
            features = i3d_model(batch_data, feature_layer=5)

        features = features[0].view(B, num_crops, 1024)

        # CRITICAL CHANGE: Max Pooling instead of Mean Pooling
        # We take the maximum activation across all spatial crops.
        # This ensures that if even one crop contains the anomaly, the signal is preserved.
        features_max = features.max(dim=1)[0].cpu().numpy() 
        
        all_features.append(features_max)

    all_features = np.concatenate(all_features, axis=0)
    np.save(save_path, all_features)

    print(f"{video_name} done. Saved to {save_path}. Final Shape: {all_features.shape}")


########################################
# DRIVER
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb", type=str)
    parser.add_argument("--load_model", default="models/baseline/model_rgb.pth", type=str)
    parser.add_argument("--input_dir", default="video_frames", type=str)
    parser.add_argument("--output_dir", default="./feature_embeddings", type=str)
    
    # Default batch_size lowered to 1 to account for the massive increase in crops
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    init_model(args.load_model)

    vid_list = []
    for root, dirs, files in os.walk(args.input_dir):
        if any(f.endswith(".jpg") for f in files):
            vid_list.append(root)

    print(f"Found {len(vid_list)} videos. Starting extraction...")

    for v in vid_list:
        run(v, args.input_dir, args.output_dir, args.batch_size, args.num_workers)

    del i3d_model
    gc.collect()