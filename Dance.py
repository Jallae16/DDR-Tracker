import pandas as pd
import numpy as np
import cv2
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import warnings
import time
from collections import deque

# Suppress protobuf warnings to keep console clean
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf.symbol_database')

def load_artifacts(model_path='Model/dance_model.keras',
                   scaler_path='Model/scaler.pkl',
                   label_encoder_path='Model/label_encoder.pkl',
                   max_seq_length_path='Model/max_seq_length.pkl'):
    """
    Load the trained Keras model and preprocessing objects.
    """
    print("[INFO] Loading model and preprocessing artifacts...")
    try:
        model = load_model(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(label_encoder_path)
        max_seq_length = joblib.load(max_seq_length_path)
    except Exception as e:
        print(f"[ERROR] Failed to load artifacts: {e}")
        exit()
    print("[INFO] Model and preprocessing artifacts loaded successfully.")
    return model, scaler, label_encoder, max_seq_length

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

def main():
    # Load the model and preprocessing tools
    model, scaler, label_encoder, max_seq_length = load_artifacts()

    # Get the list of dance labels
    dance_labels = label_encoder.classes_

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

    # Initialize frame buffer
    frame_buffer = deque(maxlen=max_seq_length)

    # Define the number of landmarks expected (MediaPipe Pose has 33)
    num_landmarks = 33
    features_per_landmark = 4  # x, y, z, visibility
    expected_features = num_landmarks * features_per_landmark  # 132

    # Game variables
    current_state = 'WAITING_TO_START'
    state_start_time = time.time()
    rounds_played = 0
    total_score = 0
    max_rounds = 3
    current_dance = None
    performance_scores = []

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
            frame_buffer.append(landmarks)
            # Visualize landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        else:
            # Optionally, you can handle missing landmarks here
            pass

        # Get the current time
        current_time = time.time()
        key = cv2.waitKey(1) & 0xFF

        # Game state management
        if current_state == 'WAITING_TO_START':
            # Display message to start the game
            cv2.putText(frame, 'Press Spacebar to start', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            if key == ord(' '):
                current_state = 'GET_READY'
                state_start_time = current_time
                print("[INFO] Starting game.")
                rounds_played = 0
                total_score = 0

        elif current_state == 'GET_READY':
            time_elapsed = current_time - state_start_time
            time_remaining = int(5 - time_elapsed)
            cv2.putText(frame, 'Get Ready!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Starting in: {time_remaining}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            if time_elapsed >= 5:
                current_state = 'PERFORMING'
                state_start_time = current_time
                # Select a random dance
                current_dance = np.random.choice(dance_labels)
                print(f"[INFO] Perform the dance: {current_dance}")
                # Reset performance scores and frame buffer
                performance_scores = []
                frame_buffer.clear()

        elif current_state == 'PERFORMING':
            time_elapsed = current_time - state_start_time
            time_remaining = int(10 - time_elapsed)
            cv2.putText(frame, f'Perform: {current_dance}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Time left: {time_remaining}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Make predictions if enough frames are available
            min_frames_for_prediction = 10
            if len(frame_buffer) >= min_frames_for_prediction:
                # Prepare the sequence
                sequence = list(frame_buffer)
                num_frames = len(sequence)

                # If sequence is shorter than max_seq_length, pad with zeros
                if num_frames < max_seq_length:
                    padding = [[0.0] * expected_features] * (max_seq_length - num_frames)
                    sequence = padding + sequence  # Pre-pad with zeros
                elif num_frames > max_seq_length:
                    sequence = sequence[-max_seq_length:]  # Take the last max_seq_length frames

                # Convert to numpy array
                sequence = np.array(sequence)  # Shape: (max_seq_length, features)

                # Scale the features
                try:
                    sequence_scaled = scaler.transform(sequence)
                except Exception as e:
                    print(f"[ERROR] Scaling failed: {e}")
                    sequence_scaled = np.zeros_like(sequence)

                # Compute variance features
                variance = np.var(sequence_scaled, axis=0).reshape(1, -1)

                # Expand dimensions for batch size
                sequence_scaled = np.expand_dims(sequence_scaled, axis=0)  # Shape: (1, max_seq_length, features)

                # Make prediction
                try:
                    prediction = model.predict([sequence_scaled, variance], verbose=0)
                    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
                    confidence = np.max(prediction)
                    print(confidence)
                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")
                    predicted_label = ["Unknown"]
                    confidence = 0.0

                # If the predicted label matches current_dance, store the confidence
                if predicted_label[0] == current_dance:
                    performance_scores.append(confidence)

            if time_elapsed >= 10:
                current_state = 'SCORING'
                state_start_time = current_time

        elif current_state == 'SCORING':
            # Compute the average confidence
            if performance_scores:
                average_confidence = sum(performance_scores) / len(performance_scores)
                score = average_confidence * 100  # Scale score as needed
                total_score += score
                print(f"[SCORE] You scored {score:.2f} points for {current_dance}")
            else:
                score = 0
                print(f"[SCORE] You scored {score:.2f} points for {current_dance}")

            # Display the score on the frame
            cv2.putText(frame, f'Scored {score:.2f} points for {current_dance}', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 255), 2, cv2.LINE_AA)

            # Wait for 2 seconds before moving to the next state
            if current_time - state_start_time >= 2:
                rounds_played += 1
                if rounds_played >= max_rounds:
                    current_state = 'GAME_OVER'
                    state_start_time = current_time
                else:
                    current_state = 'GET_READY'
                    state_start_time = current_time

        elif current_state == 'GAME_OVER':
            cv2.putText(frame, f'Game Over!', (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Final Score: {total_score:.2f}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f'Press q to quit', (10, 160), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 2, cv2.LINE_AA)
            if key == ord('q'):
                print("[INFO] Exiting...")
                break

        # Display the resulting frame
        cv2.imshow('Dance Move Detection', frame)

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
