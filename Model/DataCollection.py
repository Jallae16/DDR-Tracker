import cv2
import mediapipe as mp
import pandas as pd
import os
import time
from tqdm import tqdm  # For progress bars

# Initialize MediaPipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to create a dictionary with landmark data
def create_landmark_dict(landmarks, sample_number, frame_number, dance_name):
    landmark_dict = {
        'sample_number': sample_number,
        'frame_number': frame_number,
        'dance_name': dance_name
    }
    for idx, landmark in enumerate(landmarks.landmark):
        landmark_dict[f'landmark_{idx}_x'] = landmark.x
        landmark_dict[f'landmark_{idx}_y'] = landmark.y
        landmark_dict[f'landmark_{idx}_z'] = landmark.z
        landmark_dict[f'landmark_{idx}_visibility'] = landmark.visibility
    return landmark_dict

def process_video(video_path, dance_name, sample_number, frames_per_sample=96):
    """
    Processes a single video file to extract pose landmarks.

    Args:
        video_path (str): Path to the video file.
        dance_name (str): Name of the dance corresponding to the video.
        sample_number (int): Identifier for the sample.
        frames_per_sample (int): Number of frames to extract per sample.

    Returns:
        List[dict]: A list of dictionaries containing pose landmarks for each frame.
    """
    data = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // frames_per_sample)  # To evenly sample frames

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        frame_count = 0
        sampled_frames = 0

        with tqdm(total=frames_per_sample, desc=f"Processing Sample {sample_number}") as pbar:
            while cap.isOpened() and sampled_frames < frames_per_sample:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    # Process this frame
                    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(image)

                    if results.pose_landmarks:
                        landmark_dict = create_landmark_dict(
                            results.pose_landmarks, sample_number, sampled_frames + 1, dance_name)
                        data.append(landmark_dict)
                    else:
                        # If no landmarks detected, you can choose to skip or handle accordingly
                        pass

                    sampled_frames += 1
                    pbar.update(1)

                frame_count += 1

    cap.release()
    return data

def main():
    # User inputs
    video_directory = 'sample_data'
    output_directory = input("Enter the path to save the CSV datasets (e.g., 'dance_datasets/'): ").strip()

    # Parameters
    frames_per_sample = 96  # Fixed as per requirement

    # Ensure output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Verify video directory existence
    if not os.path.exists(video_directory):
        print(f"Error: Video directory '{video_directory}' not found.")
        exit()

    # List all video files directly inside the video_directory
    video_files = [
        f for f in os.listdir(video_directory)
        if f.endswith(('.mp4', '.avi', '.mov')) and os.path.isfile(os.path.join(video_directory, f))
    ]

    print(f"Found {len(video_files)} video(s) in '{video_directory}'.")

    # Initialize data storage
    all_data = []
    sample_number = 1

    # Process each video file
    for video_file in video_files:
        video_path = os.path.join(video_directory, video_file)
        print(f"Processing Video: {video_file} as Sample {sample_number}")

        # Replace `process_video` with your actual video processing logic
        sample_data = process_video(video_path, video_file, sample_number, frames_per_sample=frames_per_sample)

        if sample_data:
            all_data.extend(sample_data)
            print(f"Sample {sample_number} processed with {len(sample_data)} frames.")
        else:
            print(f"No pose data detected in Sample {sample_number}. Skipping.")

        sample_number += 1

    # Save data to CSV if any data was collected
    if all_data:
        df = pd.DataFrame(all_data)
        csv_filename = f"{output_directory}/all_samples.csv"
        df.to_csv(csv_filename, index=False)
        print(f"Saved data to '{csv_filename}'.")
    else:
        print("No data collected from videos. Skipping CSV creation.")

    print("\nAll videos processed successfully.")


if __name__ == "__main__":
    main()