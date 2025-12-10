import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d
import warnings

warnings.filterwarnings("ignore")


def test(
    net, test_loader, wind, test_info, step, snippet_frame_count=16, model_file=None
):
    with torch.no_grad():
        net.eval()
        net.flag = "Test"

        if model_file is not None and isinstance(model_file, str):
            net.load_state_dict(torch.load(model_file))

        frame_gt = np.load("frame_label/frame_gt.npy")
        frame_predict = []
        cls_label = []
        cls_pre = []

        for _data, _label, _name in test_loader:
            _data = _data.cuda()
            _label = _label.cuda()

            # Forward pass
            res = net(_data)
            a_predict = res["frame"].cpu().numpy()

            # --- NOTE: Ensure a_predict is 1D array of scores per snippet ---
            # If batch_size=1, a_predict might be shape (1, 32). mean(0) makes it (32,)
            video_pred = a_predict.mean(0)

            # Save classification prediction (Video level)
            cls_pre.append(1 if video_pred.max() > 0.5 else 0)

            # NOTE: You seem to be missing cls_label.append() here.
            # If _label contains the video-level label (0 or 1), you should add:
            # cls_label.append(_label.item())

            # Map snippet predictions to frame-level
            fpre_ = np.repeat(video_pred, snippet_frame_count)
            frame_predict.extend(fpre_)

        # Turn list into array
        frame_predict = np.array(frame_predict)

        # ---------------------------------------------------------
        # 2. THIS IS THE FIX (Replace the old truncation line)
        # ---------------------------------------------------------
        if len(frame_predict) != len(frame_gt):
            x_old = np.linspace(0, 1, num=len(frame_predict))
            x_new = np.linspace(0, 1, num=len(frame_gt))
            f = interp1d(x_old, frame_predict, kind="linear")
            frame_predict = f(x_new)
        # ---------------------------------------------------------

        # Compute metrics (Now lengths are guaranteed to match)
        fpr, tpr, _ = roc_curve(frame_gt, frame_predict)
        auc_score = auc(fpr, tpr)

        # WARNING: cls_label is currently empty in your code snippet.
        # You need to fix that loop above or this line will error/return NaN.
        if len(cls_label) > 0:
            accuracy = np.mean(np.array(cls_label) == np.array(cls_pre))
        else:
            accuracy = 0.0  # Placeholder to prevent crash

        precision, recall, _ = precision_recall_curve(frame_gt, frame_predict)
        ap_score = auc(recall, precision)

        wind.plot_lines("roc_auc", auc_score)
        wind.plot_lines("accuracy", accuracy)
        wind.plot_lines("pr_auc", ap_score)

        # Visualize scores (Optional: downsample for faster plotting)
        wind.lines("scores", frame_predict)
        wind.lines("roc_curve", tpr, fpr)

        test_info["step"].append(step)
        test_info["auc"].append(auc_score)
        test_info["ap"].append(ap_score)
        test_info["ac"].append(accuracy)
