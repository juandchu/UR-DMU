import os
import glob
import cv2
from multiprocessing import Pool, Value, Lock

# Configuration
SRC_DIR = "data"
OUT_DIR = "video_frames"
NUM_WORKERS = 10

# Thread-safe counter for global progress
counter = None


def init_pool(c):
    global counter
    counter = c


def dump_frames(vid_path):
    rel_path = os.path.relpath(vid_path, SRC_DIR)
    # print(rel_path)
    out_full_path = os.path.join(OUT_DIR, os.path.dirname(rel_path))

    # 1. Skip if already extracted
    if os.path.exists(out_full_path) and len(os.listdir(out_full_path)) > 0:
        with counter.get_lock():
            counter.value += 1
        print(f"[SKIPPED] Already exists: {vid_path}")
        return

    os.makedirs(out_full_path, exist_ok=True)

    # 2. Start signal
    print(f"[STARTING] Processing: {vid_path}")

    vr = cv2.VideoCapture(vid_path)
    i = 0
    while True:
        ret, frame = vr.read()
        if not ret:
            break
        cv2.imwrite(f"{out_full_path}/frame_{i:06d}.jpg", frame)
        i += 1

    vr.release()

    # 3. Completion signal with global percentage
    with counter.get_lock():
        counter.value += 1
        current_val = counter.value

    # Assuming we pass total count in a more complex way or just use a global
    print(f"[FINISHED] {current_val} videos done | Frames: {i} | Path: {vid_path}")


if __name__ == "__main__":
    video_files = glob.glob(os.path.join(SRC_DIR, "**/*.mp4"), recursive=True)
    total_videos = len(video_files)
    print(f"Total videos found: {total_videos}")

    # Shared integer across processes to track progress
    shared_counter = Value("i", 0)

    # 2. Process in parallel
    # We use init_pool to share the counter safely
    with Pool(NUM_WORKERS, initializer=init_pool, initargs=(shared_counter,)) as pool:
        pool.map(dump_frames, video_files)

    print("\n--- All extractions complete ---")
