import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import time
import random

def load_pose_data(csv_file_path='Model/dance_dataset/all_samples.csv'):
    """
    Load pose data from a CSV file.
    Returns a dictionary mapping dance sequences (sample_numbers) to lists of frames.
    Each frame is a list of 132 landmark values.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        # Verify that the expected columns are present
        expected_landmark_columns = 33 * 4  # 33 landmarks * 4 features (x, y, z, visibility)
        total_columns = df.shape[1]
        # The first three columns are 'sample_number', 'frame_number', 'dance_name'
        # So landmark columns start from index 3
        landmark_columns = df.columns[3:]  # Adjust this if you have more or fewer extra columns
        if len(landmark_columns) != expected_landmark_columns:
            print(f"[ERROR] Expected {expected_landmark_columns} landmark columns, but found {len(landmark_columns)}.")
            exit()
        # Group the data by 'sample_number' (each dance sequence)
        dance_sequences = {}
        for sample_number, group in df.groupby('sample_number'):
            # Sort the group by 'frame_number'
            group = group.sort_values(by='frame_number')
            # Extract landmark data
            frames = group[landmark_columns].values.tolist()
            dance_sequences[sample_number] = frames
        return dance_sequences
    except Exception as e:
        print(f"[ERROR] Failed to load pose data: {e}")
        exit()

def extract_landmarks(results, num_landmarks=33):
    """
    Extract and flatten landmarks from MediaPipe's Pose results.
    Returns a flat list of landmark coordinates and visibility.
    """
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    landmark_values = []
    for lm in landmarks[:num_landmarks]:
        landmark_values.extend([lm.x, lm.y, lm.z, lm.visibility])
    return landmark_values  # Length should be num_landmarks * 4

def compute_pose_difference(pose1, pose2):
    """
    Compute the difference between two poses.
    pose1 and pose2 are lists of 132 values.
    Returns the sum of absolute differences.
    """
    pose1 = np.array(pose1)
    pose2 = np.array(pose2)
    # Compute the absolute differences
    differences = np.abs(pose1 - pose2)
    # Sum the differences
    total_difference = np.sum(differences)
    return total_difference

def draw_pose_landmarks_on_frame(landmark_list, frame, color=(0, 0, 255), visibility_threshold=0.5):
    """
    Draw pose landmarks onto the frame.

    Args:
        landmark_list (list): list of 132 values (33 landmarks * 4)
        frame (numpy array): the frame to draw on
        color (tuple): color to use for drawing (B, G, R)
        visibility_threshold (float): minimum visibility to draw the landmark
    """
    # Reshape landmarks into (33, 4)
    landmarks = np.array(landmark_list).reshape(33, 4)

    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]

    # List to hold the landmark points
    landmark_points = []

    for idx, lm in enumerate(landmarks):
        x_norm, y_norm, z, visibility = lm
        if visibility < visibility_threshold:
            landmark_points.append(None)  # Invisible landmark
        else:
            x_px = int(x_norm * frame_width)
            y_px = int(y_norm * frame_height)
            landmark_points.append((x_px, y_px))

    # Draw connections
    for connection in mp.solutions.pose.POSE_CONNECTIONS:
        start_idx, end_idx = connection
        start_point = landmark_points[start_idx]
        end_point = landmark_points[end_idx]
        if start_point is not None and end_point is not None:
            cv2.line(frame, start_point, end_point, color, 2)

    # Draw landmarks
    for point in landmark_points:
        if point is not None:
            cv2.circle(frame, point, 5, color, -1)

def main():
    # Load dance sequences from CSV
    dance_sequences = load_pose_data('Model/dance_dataset/all_samples.csv')
    if not dance_sequences:
        print("[ERROR] No dance sequences loaded.")
        return

    # Get the list of dance sequence IDs
    sample_numbers = list(dance_sequences.keys())

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    # Initialize video capture (0 for default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    # Set camera resolution to 1920x1080 (1080p)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # Define the number of landmarks expected (MediaPipe Pose has 33)
    num_landmarks = 33
    features_per_landmark = 4  # x, y, z, visibility
    expected_features = num_landmarks * features_per_landmark  # 132

    # Game variables
    current_state = 'INSTRUCTIONS'
    state_start_time = time.time()
    rounds_played = 0
    total_score = 0
    max_rounds = 3
    current_dance_sequence = None
    current_dance_frame_index = 0
    total_frames_in_dance = 0
    performance_differences = []

    total_dance_duration = 10  # seconds
    time_per_dance_frame = 0   # will be computed when a dance is selected

    print("[INFO] Starting video capture. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Flip the frame horizontally for a mirror view
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB before processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        # Extract landmarks
        landmarks = extract_landmarks(results, num_landmarks=num_landmarks)
        if landmarks:
            # Visualize user's landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )

        # Get the current time
        current_time = time.time()
        key = cv2.waitKey(1) & 0xFF

        # Game state management
        if current_state == 'INSTRUCTIONS':
            # Display instructions on the frame
            cv2.putText(frame, 'Welcome to the Dance Game!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press SPACEBAR to start.', (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, 'Press Q to quit at any time.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

            if key == ord(' '):
                current_state = 'COUNTDOWN'
                state_start_time = current_time
                print("[INFO] Spacebar pressed. Starting countdown.")

        elif current_state == 'COUNTDOWN':
            time_elapsed = current_time - state_start_time
            countdown_time = 5 - int(time_elapsed)
            if countdown_time > 0:
                cv2.putText(frame, f'Starting in: {countdown_time}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                            2, (0, 255, 0), 3, cv2.LINE_AA)
            else:
                current_state = 'GET_READY'
                state_start_time = current_time
                print("[INFO] Countdown finished. Get ready for the dance.")

        elif current_state == 'GET_READY':
            # Brief pause before performing
            cv2.putText(frame, 'Get Ready!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        2, (0, 255, 0), 3, cv2.LINE_AA)
            if current_time - state_start_time >= 1:
                current_state = 'PERFORMING'
                state_start_time = current_time
                # Select a random dance sequence
                current_dance_sequence_id = random.choice(sample_numbers)
                current_dance_sequence = dance_sequences[current_dance_sequence_id]
                total_frames_in_dance = len(current_dance_sequence)
                current_dance_frame_index = 0
                # Compute time per dance frame
                total_dance_duration = 10  # seconds, adjust as needed
                time_per_dance_frame = total_dance_duration / total_frames_in_dance
                print(f"[INFO] Perform the dance sequence. Total frames: {total_frames_in_dance}")
                # Reset performance differences
                performance_differences = []

        elif current_state == 'PERFORMING':
            time_elapsed = current_time - state_start_time
            if time_elapsed >= total_dance_duration:
                # Dance sequence is over
                current_state = 'SCORING'
                state_start_time = current_time
            else:
                # Compute the current frame index
                current_dance_frame_index = int(time_elapsed / time_per_dance_frame)
                if current_dance_frame_index >= total_frames_in_dance:
                    current_dance_frame_index = total_frames_in_dance - 1  # Ensure index is within bounds

                time_remaining = int(total_dance_duration - time_elapsed + 1)
                cv2.putText(frame, 'Perform the Dance!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1.5, (0, 255, 0), 3, cv2.LINE_AA)
                cv2.putText(frame, f'Time Left: {time_remaining}s', (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2, cv2.LINE_AA)

                # Get the current dance frame
                current_dance_frame = current_dance_sequence[current_dance_frame_index]

                # Draw the dance pose onto the frame
                draw_pose_landmarks_on_frame(current_dance_frame, frame, color=(0, 0, 255))

                if landmarks:
                    # Compute the difference between user's pose and the dance pose
                    difference = compute_pose_difference(landmarks, current_dance_frame)
                    performance_differences.append(difference)

        elif current_state == 'SCORING':
            # Compute the average difference
            if performance_differences:
                average_difference = sum(performance_differences) / len(performance_differences)
                score = 100 - (average_difference * 0.01)  # Adjust as needed
                score = max(score, 0)  # Ensure the score isn't negative
                total_score += score
                print(f"[SCORE] You scored {score:.2f} points this round.")
            else:
                score = 0
                print(f"[SCORE] You scored {score:.2f} points this round.")

            # Display the score on the frame
            cv2.putText(frame, f'Scored {score:.2f} points!', (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 255, 255), 3, cv2.LINE_AA)

            # Wait for 2 seconds before moving to the next state
            if current_time - state_start_time >= 2:
                rounds_played += 1
                if rounds_played >= max_rounds:
                    current_state = 'GAME_OVER'
                    state_start_time = current_time
                else:
                    current_state = 'INSTRUCTIONS'
                    state_start_time = current_time

        elif current_state == 'GAME_OVER':
            cv2.putText(frame, f'Game Over!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1.5, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f'Final Score: {total_score:.2f}', (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0, 0, 255), 3, cv2.LINE_AA)
            cv2.putText(frame, f'Press Q to quit.', (10, 200), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            if key == ord('q'):
                print("[INFO] Exiting...")
                break

        # Display the resulting frame
        cv2.imshow('Dance Game', frame)

        # Break the loop on 'q' key press
        if key == ord('q'):
            print("[INFO] Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
