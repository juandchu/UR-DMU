import os
import glob
import cv2
from multiprocessing import Pool, Value

# Configuration
SRC_DIR = "data"
OUT_DIR = "video_frames"
NUM_WORKERS = 1  # increase if you want parallelism

counter = None

def init_pool(c):
    global counter
    counter = c

def dump_frames(vid_path: str) -> None:
    rel_path = os.path.relpath(vid_path, SRC_DIR)
    parts = rel_path.split(os.sep)

    if len(parts) < 2:
        print(f"[SKIP] Unexpected path (need at least split/video): {vid_path}")
        with counter.get_lock():
            counter.value += 1
        return

    split = parts[0]  # "train" or "test"
    video_stem = os.path.splitext(os.path.basename(vid_path))[0]

    # Force required output structure:
    # test/<video_name>/
    # train/normal/<video_name>/
    # train/abnormal/<video_name>/
    if split == "test":
        out_full_path = os.path.join(OUT_DIR, "test", video_stem)

    elif split == "train":
        if parts[1] == "normal":
            out_full_path = os.path.join(OUT_DIR, "train", "normal", video_stem)
        else:
            out_full_path = os.path.join(OUT_DIR, "train", "abnormal", video_stem)

    else:
        print(f"[SKIP] Unexpected split '{split}' in path: {vid_path}")
        with counter.get_lock():
            counter.value += 1
        return

    # Skip if already extracted
    if os.path.exists(out_full_path) and any(os.scandir(out_full_path)):
        with counter.get_lock():
            counter.value += 1
            current_val = counter.value
        print(f"[SKIPPED] {current_val} done | Already exists: {vid_path}")
        return

    os.makedirs(out_full_path, exist_ok=True)

    print(f"[STARTING] {vid_path}")

    vr = cv2.VideoCapture(vid_path)
    if not vr.isOpened():
        print(f"[ERROR] Could not open video: {vid_path}")
        with counter.get_lock():
            counter.value += 1
        return

    i = 0
    while True:
        ret, frame = vr.read()
        if not ret:
            break
        frame_path = os.path.join(out_full_path, f"frame_{i:06d}.jpg")
        ok = cv2.imwrite(frame_path, frame)
        if not ok:
            print(f"[WARN] Failed to write frame: {frame_path}")
        i += 1

    vr.release()

    with counter.get_lock():
        counter.value += 1
        current_val = counter.value

    print(f"[FINISHED] {current_val} videos done | Frames: {i} | Path: {vid_path}")

if __name__ == "__main__":
    video_files = glob.glob(os.path.join(SRC_DIR, "**", "*.mp4"), recursive=True)
    total_videos = len(video_files)
    print(f"Total videos found: {total_videos}")

    shared_counter = Value("i", 0)

    with Pool(NUM_WORKERS, initializer=init_pool, initargs=(shared_counter,)) as pool:
        pool.map(dump_frames, video_files)

    print("\n--- All extractions complete ---")