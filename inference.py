import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from scipy.interpolate import interp1d

from options import parse_args
from config import Config
from model import WSAD
from dataset_loader import FeatureDataset
# from test_function import test
import sys



# ==========================================================
# Standalone Frame-Level Inference (Reuses test logic)
# ==========================================================
def run_inference_with_test(net, test_loader, args):
    """
    Run inference on test_loader and return frame-level predictions + GT
    """
    net.eval()
    net.flag = "Test"

    all_frame_predicts = []
    all_frame_gts = []

    with torch.no_grad():
        for _data, _label, _name in test_loader:

            _data = _data.cuda()
            res = net(_data)

            # Snippet-level scores
            a_predict = res["frame"].cpu().numpy()
            video_pred = a_predict.mean(0)
            num_snippets = len(video_pred)

            # Video name
            vid_name = _name[0] if isinstance(_name, (list, tuple)) else _name
            vid_name = vid_name.replace("_i3d.npy", "")

            # Ground-truth CSV
            csv_path = os.path.join("ground_truth", vid_name, "labels.csv")
            estimated_frames = (num_snippets - 1) * args.frequency + args.chunk_size

            gt_intervals = []
            max_gt_frame = 0
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                for _, row in df.iterrows():
                    start, end = int(row["start"]), int(row["end"])
                    gt_intervals.append((start, end))
                    max_gt_frame = max(max_gt_frame, end)

            total_frames = max(estimated_frames, max_gt_frame)
            frame_gt = np.zeros(total_frames, dtype=int)
            for start, end in gt_intervals:
                frame_gt[start:end] = 1

            # Interpolate snippet scores to frame level
            x_old = np.linspace(0, 1, num=num_snippets)
            x_new = np.linspace(0, 1, num=total_frames)
            interpolator = interp1d(x_old, video_pred, kind="linear")
            frame_predict = interpolator(x_new)

            all_frame_predicts.extend(frame_predict)
            all_frame_gts.extend(frame_gt)

    return np.array(all_frame_predicts), np.array(all_frame_gts)

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    
    AUC_GRAPH_DIR = "./auc/v1-baseline/"

    # Safely INJECT your hardcoded paths into the terminal command arguments.
    # This preserves your --frequency and --chunk_size terminal inputs!
    if "--root_dir" not in sys.argv:
        sys.argv.extend(["--root_dir", "./feature_embeddings/v1-baseline/"])
        
    if "--model_path" not in sys.argv:
        sys.argv.extend(["--model_path", "./models/trained/v1-baseline/ur_dmu_best2022.pkl"])

    # ----------------------------
    # 1. Parse args & config
    # ----------------------------
    # Now this will see BOTH your terminal inputs and your injected paths above
    args = parse_args()
    config = Config(args)

    # ----------------------------
    # 2. Device
    # ----------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # ----------------------------
    # 3. Load model
    # ----------------------------
    net = WSAD(config.len_feature, flag="Test", a_nums=60, n_nums=60).to(device)

    # Using the exact file path you passed above
    model_path = config.model_path
    net.load_state_dict(torch.load(model_path, map_location=device))

    # ----------------------------
    # 4. Load test dataset
    # ----------------------------
    test_loader = DataLoader(
        FeatureDataset(
            data_dir=config.root_dir,
            mode="Test",
            modal=config.modal,
            num_segments=config.num_segments,
            len_feature=config.len_feature,
        ),
        batch_size=1,
        shuffle=False,
        num_workers=config.num_workers,
    )

    # ----------------------------
    # 5. Run inference
    # ----------------------------
    # Notice we pass 'args' directly here, so args.frequency is safely accessible 
    # even if config.py doesn't capture it!
    frame_preds, frame_gts = run_inference_with_test(net, test_loader, args)

    # ----------------------------
    # 6. Compute metrics
    # ----------------------------
    fpr, tpr, _ = roc_curve(frame_gts, frame_preds)
    roc_auc = auc(fpr, tpr)
    precision, recall, _ = precision_recall_curve(frame_gts, frame_preds)
    ap_score = auc(recall, precision)

    print("\n==============================")
    print(f"Frame-level ROC AUC : {roc_auc:.4f}")
    print(f"Frame-level AP      : {ap_score:.4f}")
    print("==============================")

    # ----------------------------
    # 7. Plot ROC curve
    # ----------------------------
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Frame-Level ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # --- Save the plot directly ---
    os.makedirs(AUC_GRAPH_DIR, exist_ok=True) 
    save_file = os.path.join(AUC_GRAPH_DIR, "roc_curve.png")
    
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"ROC Curve saved successfully to: {save_file}")

