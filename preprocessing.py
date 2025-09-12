import cv2
import time
import os
import numpy as np
from rtmlib import Wholebody
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def initialize_model(device='cuda', backend='onnxruntime'):
    """Initialize the RTMLib Wholebody model"""
    print("Initializing pose estimator with RTMLib...")
    wholebody = Wholebody(
        to_openpose=False,
        mode='balanced',
        backend=backend,
        device=device
    )
    print("Initialization successful!")
    return wholebody

def process_frame(wholebody, frame_idx, frame_rgb, width, height):
    """Process a single frame to extract pose keypoints"""
    start_time = time.time()
    keypoints, scores = wholebody(frame_rgb)
    process_time = time.time() - start_time

    if keypoints is not None:
        keypoints = keypoints / np.array([width, height])
    return frame_idx, keypoints, scores, process_time

def extract_keypoints_from_video(video_path, device='cuda', backend='onnxruntime', max_workers=4):
    """
    Extract keypoints from a video file.

    Args:
        video_path: Path to the video file
        device: Device to use for inference ('cuda' or 'cpu')
        backend: Backend to use for inference
        max_workers: Number of worker threads

    Returns:
        Dictionary containing keypoints, scores, and width/height
    """
    try:
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Initialize the model
        wholebody = initialize_model(device, backend)

        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {video_path}")

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video info: {width}x{height}, {total_frames} frames")

        # Read all frames first
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = np.ascontiguousarray(np.uint8(frame)[..., ::-1])
            frames.append((i, frame_rgb))

        cap.release()

        # Prepare output data structure
        data = {
            'keypoints': [None] * len(frames),
            'scores': [None] * len(frames),
            'w_h': [width, height]
        }

        processing_times = []

        # Process frames concurrently
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_frame, wholebody, idx, frgb, width, height) for idx, frgb in frames]

            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting keypoints"):
                idx, keypoints, scores, proc_time = future.result()
                data['keypoints'][idx] = keypoints
                data['scores'][idx] = scores
                processing_times.append(proc_time)

        # Print statistics
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            avg_fps = 1.0 / avg_time
            print(f"\nProcessing complete!")
            print(f"Average processing time: {avg_time:.3f} seconds per frame")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Total frames processed: {len(processing_times)}")

        return data

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise