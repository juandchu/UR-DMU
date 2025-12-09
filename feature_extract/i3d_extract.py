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


def load_frame(frame_file):
    data = Image.open(frame_file)
    data = data.resize((340, 256), Image.Resampling.LANCZOS)

    data = np.array(data)
    data = data.astype(float)
    data = (data * 2 / 255) - 1

    assert data.max() <= 1.0
    assert data.min() >= -1.0

    return data


def oversample_data(data):  # (39, 16, 224, 224, 2)  # 10 crop

    data_flip = np.array(data[:, :, :, ::-1, :])

    data_1 = np.array(data[:, :, :224, :224, :])
    data_2 = np.array(data[:, :, :224, -224:, :])
    data_3 = np.array(data[:, :, 16:240, 58:282, :])
    data_4 = np.array(data[:, :, -224:, :224, :])
    data_5 = np.array(data[:, :, -224:, -224:, :])

    data_f_1 = np.array(data_flip[:, :, :224, :224, :])
    data_f_2 = np.array(data_flip[:, :, :224, -224:, :])
    data_f_3 = np.array(data_flip[:, :, 16:240, 58:282, :])
    data_f_4 = np.array(data_flip[:, :, -224:, :224, :])
    data_f_5 = np.array(data_flip[:, :, -224:, -224:, :])

    return [
        data_1,
        data_2,
        data_3,
        data_4,
        data_5,
        data_f_1,
        data_f_2,
        data_f_3,
        data_f_4,
        data_f_5,
    ]


def load_rgb_batch(frames_dir, rgb_files, frame_indices):
    batch_data = np.zeros(frame_indices.shape + (256, 340, 3))
    for i in range(frame_indices.shape[0]):
        for j in range(frame_indices.shape[1]):
            batch_data[i, j, :, :, :] = load_frame(
                os.path.join(frames_dir, rgb_files[frame_indices[i][j]])
            )
    return batch_data


###---------------I3D model to extract snippet feature---------------------
# Input:  bx3x16x224x224
# Output: bx1024

i3d_model = None  # global model


def init_model(load_model_path):
    global i3d_model
    i3d_model = I3D(400, modality="rgb", dropout_prob=0, name="inception")
    i3d_model.eval()
    state = torch.load(load_model_path)
    i3d_model.load_state_dict(state)
    i3d_model.cuda()
    # keep eval() and no grad during usage
    i3d_model.eval()


def forward_batch(b_data):
    # uses global i3d_model
    global i3d_model
    b_data = b_data.transpose([0, 4, 1, 2, 3])  # to N,C,T,H,W
    b_data = torch.from_numpy(b_data)  # bx3x16x224x224
    with torch.no_grad():
        b_data = Variable(b_data.cuda()).float()
        b_features = i3d_model(b_data, feature_layer=5)
    b_features = b_features[0].data.cpu().numpy()[:, :, 0, 0, 0]
    return b_features


def run(video_dir, output_dir, batch_size):
    # note: load_model removed from here to ensure identical numeric outputs
    mode = "rgb"
    chunk_size = 16
    frequency = 16
    sample_mode = "oversample"
    video_name = video_dir.split("/")[-1]
    save_file = "{}_{}.npy".format(video_name, "i3d")
    if save_file in os.listdir(os.path.join(output_dir)):
        print("{} has been extracted".format(save_file))
        return

    rgb_files = [i for i in os.listdir(video_dir) if i.endswith("jpg")]
    rgb_files.sort(key=lambda x: int(x.split("_")[1].split(".")[0]))
    frame_cnt = len(rgb_files)

    # assert frame_cnt > chunk_size   //remove assert that checked if there were less than 16 rfames in a video. replaced for a skip

    if frame_cnt <= chunk_size:
        print(f"{video_name}: only {frame_cnt} frames. Skipping.")
        return

    clipped_length = math.ceil(frame_cnt / chunk_size)
    copy_length = (clipped_length * frequency) - frame_cnt
    if copy_length != 0:
        copy_img = [rgb_files[frame_cnt - 1]] * copy_length
        rgb_files = rgb_files + copy_img

    frame_indices = []  # Frames to chunks
    for i in range(clipped_length):
        frame_indices.append(
            [j for j in range(i * frequency, i * frequency + chunk_size)]
        )

    frame_indices = np.array(frame_indices)
    chunk_num = frame_indices.shape[0]

    batch_num = int(np.ceil(chunk_num / batch_size))
    frame_indices = np.array_split(frame_indices, batch_num, axis=0)

    full_features = [[] for _ in range(10)]

    for batch_id in tqdm(range(batch_num)):
        batch_data = load_rgb_batch(video_dir, rgb_files, frame_indices[batch_id])
        batch_data_ten_crop = oversample_data(batch_data)
        for i in range(10):
            assert batch_data_ten_crop[i].shape[-2] == 224
            assert batch_data_ten_crop[i].shape[-3] == 224
            # forward_batch now uses the single global model
            full_features[i].append(forward_batch(batch_data_ten_crop[i]))

        # free the in-batch arrays early to keep peak memory lower
        del batch_data
        del batch_data_ten_crop
        gc.collect()
        torch.cuda.empty_cache()

    full_features = [np.concatenate(i, axis=0) for i in full_features]
    full_features = [np.expand_dims(i, axis=0) for i in full_features]
    full_features = np.concatenate(full_features, axis=0)
    np.save(os.path.join(output_dir, save_file), full_features)

    # print summary identical to previous code
    print(
        "{} done: {} / {}, {}".format(
            video_name, frame_cnt, clipped_length, full_features.shape
        )
    )

    # explicitly free large objects and GPU cache
    del full_features
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="rgb", type=str)
    parser.add_argument("--load_model", default="model_rgb.pth", type=str)
    parser.add_argument("--input_dir", default="video_data_frames", type=str)
    parser.add_argument("--output_dir", default="feature_embeddings", type=str)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--sample_mode", default="oversample", type=str)
    parser.add_argument("--frequency", type=int, default=16)
    args = parser.parse_args()

    # ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # load model once (safe; does not change weights/behavior)
    init_model(args.load_model)

    # build vid_list exactly as before
    vid_list = []
    for videos in os.listdir(args.input_dir):
        for video in os.listdir(os.path.join(args.input_dir, videos)):
            save_file = "{}_{}.npy".format(video, "i3d")
            if save_file in os.listdir(os.path.join(args.output_dir)):
                print("{} has been extracted".format(save_file))
            else:
                vid_list.append(os.path.join(args.input_dir, videos, video))

    nums = len(vid_list)
    print("leave {} videos".format(nums))

    # run sequentially (no multiprocessing)
    for vdir in vid_list:
        run(vdir, args.output_dir, args.batch_size)

    # final cleanup
    del i3d_model
    gc.collect()
    torch.cuda.empty_cache()
