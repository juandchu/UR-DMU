import cv2
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

# Reuse your existing imports
from model import WSAD
from config import Config
from options import parse_args
from dataset_loader import FeatureDataset

import cv2
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from scipy.interpolate import interp1d

def visualize_results(net, test_loader, args, video_folder, output_dir="./visualizations"):
    os.makedirs(output_dir, exist_ok=True)
    net.eval()
    
    extensions = ['.mp4', '.avi', '.mkv', '.mov']

    for i, (data, label, name) in enumerate(test_loader):
        if i >= 3: break
        
        vid_name = name[0] if isinstance(name, (list, tuple)) else name
        vid_name = vid_name.replace("_i3d.npy", "").replace(".npy", "")
        
        # 1. Target the specific subfolder structure
        # Expected: data/test/video_eo_167/video.mp4
        video_subfolder = os.path.join(video_folder, vid_name)
        video_path = None
        
        if os.path.exists(video_subfolder):
            for ext in extensions:
                temp_path = os.path.join(video_subfolder, vid_name + ext)
                if os.path.exists(temp_path):
                    video_path = temp_path
                    break
        
        # Notification if the specific video file is missing
        if not video_path:
            print(f"FAILED: Could not find 'video.mp4' (or other formats) in {video_subfolder}")
            continue

        # 2. Run Model Inference
        with torch.no_grad():
            res = net(data.cuda())
            snippet_scores = res["frame"].cpu().numpy().squeeze() 
            if snippet_scores.ndim > 1:
                snippet_scores = snippet_scores.mean(0)
            num_snippets = len(snippet_scores)
            
        # 3. Open Video Stream
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 4. Load Ground Truth
        csv_path = os.path.join("ground_truth", vid_name, "labels.csv")
        gt_array = np.zeros(total_frames)
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            for _, row in df.iterrows():
                s, e = int(row["start"]), int(row["end"])
                # Boundary check for safety
                if s < total_frames:
                    gt_array[s:min(e, total_frames)] = 1
        else:
            print(f"⚠️ Warning: No CSV found at {csv_path}")

        # 5. Interpolate snippet scores to match EVERY video frame
        x_old = np.linspace(0, 1, num=num_snippets)
        x_new = np.linspace(0, 1, num=total_frames)
        f_interp = interp1d(x_old, snippet_scores, kind='linear', fill_value="extrapolate")
        frame_scores = f_interp(x_new)

        # 6. Setup Video Writer
        out_path = os.path.join(output_dir, f"{vid_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        print(f" SUCCESS: Processing {vid_name} ({total_frames} frames)...")

        for f_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret: break

            score = frame_scores[f_idx]
            is_anomaly = gt_array[f_idx]

            # --- DRAWING OVERLAYS ---
            # Box for readability
            cv2.rectangle(frame, (10, 10), (400, 160), (0, 0, 0), -1)
            
            # Anomaly Score
            cv2.putText(frame, f"Model Score: {score:.3f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Ground Truth Status
            gt_text = "GT: ANOMALY" if is_anomaly else "GT: NORMAL"
            gt_color = (0, 0, 255) if is_anomaly else (0, 255, 0)
            cv2.putText(frame, gt_text, (20, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, gt_color, 2)
            
            # Frame counter
            cv2.putText(frame, f"Frame: {f_idx}/{total_frames}", (20, 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Timeline bar at bottom
            bar_h = 20
            # Current progress pointer
            pointer_x = int((f_idx / total_frames) * width)
            cv2.rectangle(frame, (0, height-bar_h), (width, height), (50, 50, 50), -1)
            cv2.rectangle(frame, (pointer_x - 2, height-bar_h), (pointer_x + 2, height), (255, 255, 255), -1)

            out.write(frame)

        cap.release()
        out.release()

    print(f"Done! Check the '{output_dir}' folder.")

if __name__ == "__main__":
    # Setup args (same as your original script)
    args = parse_args()
    config = Config(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    net = WSAD(config.len_feature, flag="Test", a_nums=60, n_nums=60).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))

    # Load Data
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


    # PATH TO YOUR RAW VIDEOS (Change this to where your .mp4 files are)
    VIDEO_DATA_DIR = "./data/test/" 
    
    visualize_results(net, test_loader, args, VIDEO_DATA_DIR)