import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, average_precision_score
from scipy.interpolate import interp1d
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")


def test(net, test_loader, test_info, step, args, model_file=None):
    """
    Evaluates the model on the test dataset.
    Args:
        args: Must contain args.frequency and args.chunk_size passed from main.py
    """
    with torch.no_grad():
        net.eval()
        net.flag = "Test"
        device = next(net.parameters()).device
        debug_test = bool(getattr(args, "test_debug", False)) or os.environ.get("URDMU_TEST_DEBUG") == "1"

        if model_file is not None and isinstance(model_file, str):
            net.load_state_dict(torch.load(model_file, map_location=device))

        # We will collect the ground truths and predictions for ALL videos
        # to calculate a global AUC score at the very end.
        all_frame_predicts = []
        all_frame_gts = []
        videos_seen = 0
        videos_with_gt = 0

        for _data, _label, _name in test_loader:
            _data = _data.to(device)

            # --------------------------------------------------------
            # 1. Get Model Predictions
            # --------------------------------------------------------
            res = net(_data)

            # Extract snippet-level scores. Expected shape: (T,) or (1,T)
            snippet_scores = res["frame"].detach().float().cpu().numpy()
            snippet_scores = np.squeeze(snippet_scores)
            if snippet_scores.ndim > 1:
                snippet_scores = snippet_scores.mean(axis=0)
            if snippet_scores.ndim != 1:
                raise ValueError(
                    f"Unexpected prediction shape {snippet_scores.shape}; expected 1D snippet scores."
                )
            num_snippets = int(snippet_scores.shape[0])
            videos_seen += 1

            # --------------------------------------------------------
            # 2. Extract Video Name
            # --------------------------------------------------------
            # Dataloader usually returns tuples for strings. Extract the raw name.
            vid_name = (
                _name[0]
                if isinstance(_name, tuple) or isinstance(_name, list)
                else _name
            )
            # Normalize to a ground-truth folder name:
            #   "video_xxx_i3d.npy" -> "video_xxx"
            #   "video_xxx.npy"     -> "video_xxx"
            vid_name = Path(str(vid_name)).stem
            if vid_name.endswith("_i3d"):
                vid_name = vid_name[: -len("_i3d")]
            # --------------------------------------------------------
            # 3. Build the Ground Truth Array
            # --------------------------------------------------------
            csv_path = os.path.join("ground_truth", vid_name, "labels.csv")

            # Math: How long was the original video?
            # If we have 10 snippets, an overlap frequency of 8, and a chunk size of 16:
            # (10 - 1) * 8 + 16 = 88 frames
            estimated_frames = (num_snippets - 1) * args.frequency + args.chunk_size
            # print(estimated_frames)
            gt_intervals = []
            max_gt_frame = 0

            # Check if this video has a CSV (Normal videos might not have one)
            if os.path.exists(csv_path):
                videos_with_gt += 1
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    start, end = int(row["start"]), int(row["end"])
                    gt_intervals.append((start, end))
                    if end > max_gt_frame:
                        max_gt_frame = end

            # Safety Check: Make sure our estimated length isn't shorter than the last anomaly!
            total_frames = max(estimated_frames, max_gt_frame)

            # Create an array of 0s (Normal) for the entire video
            frame_gt = np.zeros(total_frames, dtype=int)

            # Inject 1s (Abnormal) based on the CSV intervals
            for start, end in gt_intervals:
                start = max(0, start)
                end = min(total_frames, end)
                if end <= start:
                    continue
                # Array slicing efficiently labels the chunks: e.g., frames 150 to 780
                frame_gt[start:end] = 1
            # --------------------------------------------------------
            # 4. Interpolate Predictions
            # --------------------------------------------------------
            # This stretches our snippet-level predictions across the total frame count.
            # It naturally smooths out overlapping chunks without using step-functions.

            if num_snippets <= 0:
                raise ValueError(f"{vid_name}: num_snippets was {num_snippets}, expected > 0.")

            if num_snippets == 1:
                frame_predict = np.full(total_frames, float(snippet_scores[0]), dtype=np.float32)
            else:
                x_old = np.linspace(0, 1, num=num_snippets, dtype=np.float32)
                x_new = np.linspace(0, 1, num=total_frames, dtype=np.float32)
                interpolator = interp1d(
                    x_old, snippet_scores, kind="linear", bounds_error=False, fill_value="extrapolate"
                )
                frame_predict = interpolator(x_new)

            # Add this video's frames to our global dataset lists
            all_frame_predicts.extend(frame_predict)
            all_frame_gts.extend(frame_gt)

        # --------------------------------------------------------
        # 5. Calculate Final Metrics
        # --------------------------------------------------------
        all_frame_predicts = np.array(all_frame_predicts)
        all_frame_gts = np.array(all_frame_gts)

        if all_frame_gts.min() == all_frame_gts.max():
            raise ValueError(
                "Cannot compute ROC AUC: only one class present in ground truth. "
                f"(videos_seen={videos_seen}, videos_with_gt={videos_with_gt})"
            )

        # Standard ROC curve calculation
        fpr, tpr, _ = roc_curve(all_frame_gts, all_frame_predicts)
        auc_score = auc(fpr, tpr)

        # Precision-Recall (Average Precision)
        ap_score = average_precision_score(all_frame_gts, all_frame_predicts)
        preds_binary = (all_frame_predicts > 0.5).astype(int)
        accuracy = (preds_binary == all_frame_gts).mean()

        # Print feedback directly to the terminal
        print(f"Step: {step} | AUC: {auc_score:.4f} | AP: {ap_score:.4f}")
        if debug_test:
            pos_rate = float(all_frame_gts.mean())
            pred_std = float(all_frame_predicts.std())
            print(
                f"[test debug] videos={videos_seen} with_gt={videos_with_gt} "
                f"pos_rate={pos_rate:.4f} pred_mean={all_frame_predicts.mean():.4f} pred_std={pred_std:.6f}"
            )

        # Update the tracking dictionary
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
