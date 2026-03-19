import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from scipy.interpolate import interp1d

from components.options import parse_args
from components.config import Config
from components.model import WSAD
from components.dataset_loader import FeatureDataset
import sys

# ==========================================================
# Standalone Frame-Level Inference 
# ==========================================================
def run_inference_with_test(net, test_loader, args, version_name):
    net.eval()
    net.flag = "Test"

    all_frame_predicts = []
    all_frame_gts = []
    
    # Store individual video results
    video_results = []

    with torch.no_grad():
        for _data, _label, _name in test_loader:
            _data = _data.cuda()
            res = net(_data)

            a_predict = res["frame"].cpu().numpy()
            video_pred = a_predict.mean(0) 
            num_snippets = len(video_pred) 

            vid_name = _name[0] if isinstance(_name, (list, tuple)) else _name
            vid_name = vid_name.replace("_i3d.npy", "")

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
            
            x_old = np.linspace(0, 1, num=num_snippets) 
            x_new = np.linspace(0, 1, num=total_frames) 
            interpolator = interp1d(x_old, video_pred, kind="linear")
            frame_predict = interpolator(x_new)

            # --- CALCULATE INDIVIDUAL VIDEO SCORE ---
            # We use AUC for the individual video to see if it separated anomaly from normal
            try:
                if np.sum(frame_gt) > 0: # Only calc AUC if there is an anomaly present
                    v_fpr, v_tpr, _ = roc_curve(frame_gt, frame_predict)
                    v_auc = auc(v_fpr, v_tpr)
                else:
                    v_auc = 0.0 # Or mark as "Normal Video"
            except:
                v_auc = 0.0

            print(f"Video: {vid_name:<30} | AUC: {v_auc:.4f}")
            video_results.append({"name": vid_name, "auc": v_auc})

            all_frame_predicts.extend(frame_predict)
            all_frame_gts.extend(frame_gt)
            
            # Plotting logic remains same...
            os.makedirs(f"debug_plots/{version_name}", exist_ok=True)
            plt.figure()
            plt.plot(frame_predict, label=f"pred (AUC:{v_auc:.2f})")
            plt.plot(frame_gt, label="GT")
            plt.title(f"{vid_name}")
            plt.savefig(f"debug_plots/{version_name}/{vid_name}.png")
            plt.close()

    # # Optional: Save a CSV of all video scores for your PPT
    df_results = pd.DataFrame(video_results)
    df_results.to_csv(f"debug_plots/auc_per_video/{version_name}_scores.csv", index=False)

    return np.array(all_frame_predicts), np.array(all_frame_gts)
# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    VERSION = "v6-4_crop_finetuned_model_newfreq" 
    AUC_GRAPH_DIR = f"./auc/{VERSION}/"

    # Safely INJECT your hardcoded paths
    if "--root_dir" not in sys.argv:
        sys.argv.extend(["--root_dir", f"./feature_embeddings/{VERSION}/"])
        
    if "--model_path" not in sys.argv:
        sys.argv.extend(["--model_path", f"./models/trained/{VERSION}/ur_dmu_best2022.pkl"])

    # 1. Parse args & config
    args = parse_args()
    config = Config(args)

    # 2. Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)

    # 3. Load model
    net = WSAD(config.len_feature, flag="Test", a_nums=config.a_nums, n_nums=config.n_nums).to(device)
    model_path = config.model_path
    net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    # 4. Load test dataset
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

    # 5. Run inference (Notice we pass VERSION here now so the function knows where to save the debug plots)
    frame_preds, frame_gts = run_inference_with_test(net, test_loader, args, VERSION)
    avg_model_score=np.mean(frame_preds)

    # ----------------------------
    # 6. Compute Base Metrics (AUC & AP)
    # ----------------------------
    # ROC Curve and AUC
    fpr, tpr, _ = roc_curve(frame_gts, frame_preds)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall Curve and AP
    precision, recall, _ = precision_recall_curve(frame_gts, frame_preds)
    ap_score = average_precision_score(frame_gts, frame_preds)

    print("\n==============================")
    print(f"Frame-level ROC AUC : {roc_auc:.4f}")
    print(f"Frame-level AP      : {ap_score:.4f}")
    print(f"Average Model Score : {avg_model_score:.4f}")
    print("==============================")

    # ----------------------------
    # 7. Plot Side-by-Side Curves
    # ----------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # --- Plot 1: ROC Curve (AUC) ---
    ax1.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax1.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right")
    ax1.grid(True)

    # --- Plot 2: Precision-Recall Curve (AP) ---
    ax2.plot(recall, precision, color="green", lw=2, label=f"AP = {ap_score:.4f}")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left")
    ax2.grid(True)

    plt.tight_layout()

    # --- Save the plot directly ---
    os.makedirs(AUC_GRAPH_DIR, exist_ok=True) 
    save_file = os.path.join(AUC_GRAPH_DIR, "auc_and_ap_curves.png")
    
    plt.savefig(save_file, dpi=300, bbox_inches='tight')
    print(f"Evaluation graphs saved successfully to: {save_file}")