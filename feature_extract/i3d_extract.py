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

CHUNK_SIZE = 16  # This is the sliding window size
FREQUENCY = 16  # How many frames after the last 'window start frame' should the next window start
MIN_FRAMES = 16  # Minimum number of frames per video to discard video


########################################
# 1. PYTORCH DATASET
########################################
class VideoSnippetDataset(Dataset):
    def __init__(self, frames_dir, rgb_files, frame_indices):
        self.frames_dir = frames_dir
        self.rgb_files = rgb_files
        self.frame_indices = frame_indices

        # Highly optimized torchvision transforms
        self.transform = T.Compose(
            [
                # PIL resize expects (W, H) but torchvision Resize expects (H, W)
                T.Resize((256, 340), interpolation=T.InterpolationMode.LANCZOS),
                T.ToTensor(),  # Converts to [0.0, 1.0] and moves channels to (C, H, W)
                T.Normalize(
                    mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
                ),  # Normalizes to [-1, 1]
            ]
        )

    def __len__(self):
        return len(self.frame_indices)

    def __getitem__(self, idx):
        indices = self.frame_indices[idx]
        snippet = []

        for frame_idx in indices:
            img_path = os.path.join(self.frames_dir, self.rgb_files[frame_idx])
            img = Image.open(img_path).convert("RGB")

            # Apply transforms
            img_tensor = self.transform(img)  # Shape: (C, 256, 340)

            # Center crop to 224x224 (H: 16 to 240, W: 58 to 282)
            img_tensor = img_tensor[:, 16:240, 58:282]
            snippet.append(img_tensor)

        # Stack into (C, T, H, W) which I3D expects
        return torch.stack(snippet, dim=1)


########################################
# 2. MODEL INITIALIZATION
########################################
i3d_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_model(load_model_path):
    global i3d_model
    i3d_model = I3D(400, modality="rgb", dropout_prob=0, name="inception").to(device)
    i3d_model.eval()

    checkpoint = torch.load(load_model_path, map_location=device)
    state = (
        checkpoint["model_state_dict"]
        if "model_state_dict" in checkpoint
        else checkpoint
    )

    # remove head so it can load into any n-way model
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
        print(f"{video_name}: only {frame_cnt} frames. Skipping…")
        return

    # Snippet logic
    T_chunks = math.ceil(frame_cnt / CHUNK_SIZE)
    pad = (T_chunks * FREQUENCY) - frame_cnt
    if pad > 0:
        rgb_files += [rgb_files[-1]] * pad

    frame_indices = np.array(
        [
            list(range(i * FREQUENCY, i * FREQUENCY + CHUNK_SIZE))
            for i in range(T_chunks)
        ]
    )

    # Initialize Dataset and DataLoader
    dataset = VideoSnippetDataset(video_dir, rgb_files, frame_indices)

    # This is where the magic happens: multiple workers fetching data in the background
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,  # Speeds up CPU to GPU memory transfer
    )

    all_features = []
    global i3d_model

    for batch_data in tqdm(dataloader, desc=video_name):
        # batch_data is already shaped (B, C, T, H, W) by the DataLoader
        batch_data = batch_data.to(device)

        with torch.no_grad():
            features = i3d_model(batch_data, feature_layer=5)

        # features[0] shape is (B, T_out, 1, 1, 1024)
        features = features[0].cpu().numpy()[:, :, 0, 0, 0]  # → (B, 1024)
        all_features.append(features)

    all_features = np.concatenate(all_features, axis=0)
    np.save(save_path, all_features)

    print(f"{video_name} done. Saved to {save_path}. Shape: {all_features.shape}")


########################################
# DRIVER
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb", type=str)
    parser.add_argument(
        "--load_model",
        default="feature_extract/models/fine_tuning/i3d_best.pth",
        type=str,
    )
    parser.add_argument("--input_dir", default="video_frames", type=str)
    parser.add_argument("--output_dir", default="./feature_embeddings", type=str)
    parser.add_argument("--batch_size", type=int, default=40)
    # Added a parameter so you can tune the CPU workers easily from the command line
    parser.add_argument("--num_workers", type=int, default=12)
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
