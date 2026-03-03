import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d
import warnings

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

        if model_file is not None and isinstance(model_file, str):
            net.load_state_dict(torch.load(model_file))

        # We will collect the ground truths and predictions for ALL videos
        # to calculate a global AUC score at the very end.
        all_frame_predicts = []
        all_frame_gts = []

        for _data, _label, _name in test_loader:
            _data = _data.cuda()

            # --------------------------------------------------------
            # 1. Get Model Predictions
            # --------------------------------------------------------
            res = net(_data)

            # Extract the raw snippet scores. Shape: (num_snippets,)
            a_predict = res["frame"].cpu().numpy()
            video_pred = a_predict.mean(0)
            num_snippets = len(video_pred)

            # --------------------------------------------------------
            # 2. Extract Video Name
            # --------------------------------------------------------
            # Dataloader usually returns tuples for strings. Extract the raw name.
            vid_name = (
                _name[0]
                if isinstance(_name, tuple) or isinstance(_name, list)
                else _name
            )
            # print(vid_name)
            vid_name = vid_name.replace(
                "_i3d.npy", ""
            )  # Clean the string (e.g., "video_01")
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
                # Array slicing efficiently labels the chunks: e.g., frames 150 to 780
                frame_gt[start:end] = 1
            # --------------------------------------------------------
            # 4. Interpolate Predictions
            # --------------------------------------------------------
            # This stretches our snippet-level predictions across the total frame count.
            # It naturally smooths out overlapping chunks without using step-functions.

            x_old = np.linspace(0, 1, num=num_snippets)
            x_new = np.linspace(0, 1, num=total_frames)

            interpolator = interp1d(x_old, video_pred, kind="linear")
            frame_predict = interpolator(x_new)

            # Add this video's frames to our global dataset lists
            all_frame_predicts.extend(frame_predict)
            all_frame_gts.extend(frame_gt)

        # --------------------------------------------------------
        # 5. Calculate Final Metrics
        # --------------------------------------------------------
        all_frame_predicts = np.array(all_frame_predicts)
        all_frame_gts = np.array(all_frame_gts)

        # Standard ROC curve calculation
        fpr, tpr, _ = roc_curve(all_frame_gts, all_frame_predicts)
        auc_score = auc(fpr, tpr)

        # Precision-Recall curve computation
        precision, recall, _ = precision_recall_curve(all_frame_gts, all_frame_predicts)
        ap_score = auc(recall, precision)
        preds_binary = (all_frame_predicts > 0.5).astype(int)
        accuracy = (preds_binary == all_frame_gts).mean()

        # Print feedback directly to the terminal
        print(f"Step: {step} | AUC: {auc_score:.4f} | AP: {ap_score:.4f}")

        # Update the tracking dictionary
        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
