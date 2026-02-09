from multiprocessing import Pool
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from tqdm import tqdm
import torch
import argparse
from PIL import Image
from torch.autograd import Variable
import numpy as np
from i3dpt import I3D
import math
import gc

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

########################################
# LOAD AND PREPROCESS FRAMES
########################################
def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.Resampling.LANCZOS)
    data = np.array(data).astype(float)
    data = (data * 2 / 255) - 1  # normalize to [-1,1]
    return data


########################################
# CENTER CROP ONLY  → produces (T,1024)
########################################
def center_crop(data):
    # data shape: (B, 16, 256, 340, 3)
    # crop to 224x224 center
    return data[:, :, 16:240, 58:282, :]  # → shape (B,16,224,224,3)


########################################
# CREATE SNIPPET BATCHES
########################################
def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]])
            )
    return batch_data


########################################
# FORWARD PASS THROUGH I3D
########################################
i3d_model = None

def init_model(load_model_path):
    global i3d_model
    i3d_model = I3D(400, modality="rgb", dropout_prob=0, name="inception").cuda()
    i3d_model.eval()

    checkpoint = torch.load(load_model_path, map_location="cpu")
    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint

    # remove finetuned head (155-way) so it can load into a 400-way model
    state = {k: v for k, v in state.items() if not k.startswith("conv3d_0c_1x1.")}

    i3d_model.load_state_dict(state, strict=False)
    i3d_model.eval()

def forward_batch(b_data):
    global i3d_model
    b_data = b_data.transpose([0, 4, 1, 2, 3])  # → N,C,T,H,W
    b_data = torch.from_numpy(b_data).float().cuda()

    with torch.no_grad():
        features = i3d_model(b_data, feature_layer=5)

    # features[0] shape is (1, T, 1, 1, 1024) or (N,T,1,1,1)
    features = features[0].cpu().numpy()[:, :, 0, 0, 0]  # → (N,1024)

    return features


########################################
# MAIN EXTRACTION FUNCTION
########################################
def run(video_dir, output_dir, batch_size):

    video_name = video_dir.split("/")[-1]
    save_file = f"{video_name}_i3d.npy"

    if save_file in os.listdir(output_dir):
        print(f"{save_file} already exists.")
        return

    rgb_files = sorted(
        [f for f in os.listdir(video_dir) if f.endswith("jpg")],
        key=lambda x: int(x.split("_")[1].split(".")[0]),
    )

    frame_cnt = len(rgb_files)
    if frame_cnt <= 16:
        print(f"{video_name}: only {frame_cnt} frames. Skipping…")
        return

    chunk_size = 16
    frequency = 16

    T = math.ceil(frame_cnt / chunk_size)

    # copy last frame to fill incomplete last chunk
    pad = (T * frequency) - frame_cnt
    if pad > 0:
        rgb_files += [rgb_files[-1]] * pad

    # prepare snippet indices
    frame_indices = np.array([
        list(range(i*frequency, i*frequency+chunk_size))
        for i in range(T)
    ])

    # batch snippet indices
    batches = np.array_split(frame_indices, math.ceil(T / batch_size))

    all_features = []

    for batch_idx, batch_inds in enumerate(tqdm(batches, desc=video_name)):

        batch_data = load_rgb_batch(video_dir, rgb_files, batch_inds)

        # center crop ONLY
        batch_data = center_crop(batch_data)

        # forward pass → shape (B,1024)
        feats = forward_batch(batch_data)

        all_features.append(feats)

        del batch_data
        gc.collect()
        torch.cuda.empty_cache()

    # final shape (T, 1024)
    all_features = np.concatenate(all_features, axis=0)

    np.save(os.path.join(output_dir, save_file), all_features)

    print(f"{video_name} done. Final feature shape: {all_features.shape}")


########################################
# DRIVER
########################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb", type=str)
    parser.add_argument("--load_model", default="feature_extract/models/fine_tuning/i3d_best.pth", type=str)
    parser.add_argument("--input_dir", default="video_data_frames", type=str)
    parser.add_argument("--output_dir", default="./feature_embeddings", type=str)
    parser.add_argument("--batch_size", type=int, default=20)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    init_model(args.load_model)

    vid_list = []
    for vid in os.listdir(args.input_dir):
        vid_path = os.path.join(args.input_dir, vid)
        if not os.path.isdir(vid_path):
            continue
        save_file = f"{vid}_i3d.npy"
        if save_file not in os.listdir(args.output_dir):
            vid_list.append(vid_path)
        print(f"Extracting {len(vid_list)} videos…")

    for v in vid_list:
        run(v, args.output_dir, args.batch_size)

    del i3d_model
    gc.collect()
