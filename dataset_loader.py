import torch
import torch.utils.data as data
import os
import numpy as np
import utils


class FeatureDataset(data.Dataset):
    def __init__(
        self, root_dir, modal, mode, num_segments, len_feature, seed=-1, is_normal=None
    ):
        # If another seed is passed
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature

        split_path = os.path.join("list", "data_{}.list".format(self.mode))
        split_file = open(split_path, "r")
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.split())
        split_file.close()

        # filter normal and abnormal
        if is_normal is not None:
            filtered_list = []
            for path in self.vid_list:
                if is_normal and "norm" in path[0].lower():
                    filtered_list.append(path)
                elif not is_normal and "ab" in path[0].lower():
                    filtered_list.append(path)
            self.vid_list = filtered_list

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        if self.mode == "Test":
            data, label, name = self.get_data(index)
            # print("Test", index, name)
            return data, label, name
        else:
            data, label = self.get_data(index)
            print("Train", index)
            return data, label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]
        filename = vid_info.split("/")[-1]  # e.g. "video_eo_94_650_ab.npy"
        name = "_".join(filename.split("_")[:3])
        # print(vid_info)
        print("this should print")
        feature_path = os.path.join("feature_embeddings", vid_info)
        if not os.path.exists(feature_path):
            print(f"Warning: {feature_path} not found. Skipping.")
            return None  # mark missing entry

        video_feature = np.load(os.path.join("feature_embeddings", vid_info)).astype(
            np.float32
        )

        f_lower = filename.lower()
        if "norm" in f_lower:
            label = 0  # normal
        else:
            label = 1  # abnormal

        # ----------------------------
        # 4. Temporal segmentation (training only)
        # ----------------------------
        if self.mode == "Train":
            new_feat = np.zeros(
                (self.num_segments, video_feature.shape[1]), dtype=np.float32
            )

            # Split frames into num_segments equal parts
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=int)

            for i in range(self.num_segments):
                start, end = r[i], r[i + 1]
                if start < end:
                    new_feat[i] = np.mean(video_feature[start:end], axis=0)
                else:
                    new_feat[i] = video_feature[start]

            video_feature = new_feat

        # ----------------------------
        # 5. Return data
        # ----------------------------
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label
