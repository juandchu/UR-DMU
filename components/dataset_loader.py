import torch
import torch.utils.data as data
import numpy as np
from pathlib import Path
import components.utils as utils


class FeatureDataset(data.Dataset):
    def __init__(
        self,
        data_dir,  # Point this to the main folder containing train/test dirs
        mode,  # "Train" or "Test"
        num_segments,  # The target size (N=200) for the video
        len_feature=1024,
        modal="RGB",
        seed=-1,
        is_normal=None,
    ):
        # 1. Reproducibility
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.num_segments = num_segments
        self.len_feature = len_feature
        self.is_normal = is_normal

        # Use pathlib for robust, OS-agnostic path building
        self.data_dir = Path(data_dir)
        self.vid_list = []

        # 2. Dynamic File Discovery based on folder routes
        if self.mode == "Train":
            # Load Normal features if requested (or if we want both)
            if is_normal is True or is_normal is None:
                norm_dir = self.data_dir / "train" / "normal"
                self.vid_list.extend(list(norm_dir.glob("*.npy")))

            # Load Abnormal features if requested (or if we want both)
            if is_normal is False or is_normal is None:
                abn_dir = self.data_dir / "train" / "abnormal"
                self.vid_list.extend(list(abn_dir.glob("*.npy")))

        elif self.mode == "Test":
            # rglob recursively searches, so it works whether test files
            # are loose in /test/ or nested in /test/normal/ etc.
            test_dir = self.data_dir / "test"
            self.vid_list.extend(list(test_dir.rglob("*.npy")))

        # Safety Check: Alert if paths were wrong or empty
        if not self.vid_list:
            raise FileNotFoundError(
                f"No .npy files found in {self.data_dir} for mode={self.mode}"
            )

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
        # 3. Locate and Load File
        file_path = self.vid_list[index]
        name = file_path.name  # Extracts just "v_123.npy" cleanly

        # Load the binary .npy file
        video_feature = np.load(file_path).astype(np.float32)

        # 4. Ground Truth Labeling
        # Prefer directory-based labels (train/normal vs train/abnormal) over filename heuristics.
        # This avoids accidentally labeling *all* files as abnormal when filenames don't contain "norm".
        label = None
        try:
            rel_parts = file_path.relative_to(self.data_dir).parts
        except ValueError:
            rel_parts = file_path.parts
        rel_parts_lower = [p.lower() for p in rel_parts]

        if "normal" in rel_parts_lower:
            label = 0
        elif "abnormal" in rel_parts_lower:
            label = 1

        if label is None:
            if self.is_normal is True:
                label = 0
            elif self.is_normal is False:
                label = 1

        if label is None:
            f_lower = name.lower()
            label = 0 if ("norm" in f_lower or "normal" in f_lower) else 1

        # ----------------------------
        # 5. Temporal segmentation (UR-DMU compression logic preserved)
        # ----------------------------
        if self.mode == "Train":
            new_feat = np.zeros(
                (self.num_segments, video_feature.shape[1]), dtype=np.float32
            )

            r = np.linspace(0, len(video_feature), self.num_segments + 1, dtype=int)

            for i in range(self.num_segments):
                start, end = r[i], r[i + 1]

                if start < end:
                    new_feat[i] = np.mean(video_feature[start:end], axis=0)
                else:
                    new_feat[i] = video_feature[start]

            video_feature = new_feat

        # 6. Return Data
        if self.mode == "Test":
            return video_feature, label, name
        else:
            return video_feature, label
