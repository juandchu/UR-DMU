import torch
import torch.utils.data as data
import os
import numpy as np
import utils


class FeatureDataset(data.DataLoader):
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

        # For generic dataset, optionally filter by is_normal if needed
        if is_normal is not None:
            filtered_list = []
            for path in self.vid_list:
                if is_normal and "normal" in path[0].lower():
                    filtered_list.append(path)
                elif not is_normal and "abnormal" in path[0].lower():
                    filtered_list.append(path)
            self.vid_list = filtered_list

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):

        if self.mode == "Test":
            data, label, name = self.get_data(index)
            return data, label, name
        else:
            data, label = self.get_data(index)
            return data, label

    def get_data(self, index):
        vid_info = self.vid_list[index][0]
        name = vid_info.split("/")[-1].split("_x264")[0]
        video_feature = np.load(vid_info).astype(np.float32)

        if "norm" in vid_info.split("/")[-1]:
            label = 0
        else:
            label = 1
        if self.mode == "Train":
            new_feat = np.zeros((self.num_segments, video_feature.shape[1])).astype(
                np.float32
            )
            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=np.int)
            for i in range(self.num_segments):
                if r[i] != r[i + 1]:
                    new_feat[i, :] = np.mean(video_feature[r[i] : r[i + 1], :], 0)
                else:
                    new_feat[i : i + 1, :] = video_feature[r[i] : r[i] + 1, :]
            video_feature = new_feat
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label
