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

    # Initialize frame buffer
    frame_buffer = deque(maxlen=max_seq_length)

    # Timing for predictions
    prediction_interval = 5  # seconds
    last_prediction_time = time.time()

    # Define the number of landmarks expected (MediaPipe Pose has 33)
    num_landmarks = 33
    features_per_landmark = 4  # x, y, z, visibility
    expected_features = num_landmarks * features_per_landmark  # 132

    # Initialize variables for displaying detected dance
    last_detected_dance = None
    last_confidence = 0.0
    dance_display_start_time = 0  # Time when the dance was detected
    SIGN_DISPLAY_DURATION = 9999    # Duration (seconds) to display detected sign (set to a large number)

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

        # Check if it's time to make a prediction
        current_time = time.time()
        if (current_time - last_prediction_time) >= prediction_interval:
            if len(frame_buffer) < max_seq_length:
                print("[WARNING] Not enough frames to make a prediction.")
            else:
                print("[INFO] Making a prediction...")
                # Prepare the sequence
                sequence = list(frame_buffer)
                num_frames = len(sequence)

                # If sequence is shorter than max_seq_length, pad with zeros
                if num_frames < max_seq_length:
                    padding = [ [0.0]*expected_features ] * (max_seq_length - num_frames)
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
                    prediction = model.predict([sequence_scaled, variance])
                    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
                    confidence = np.max(prediction)
                except Exception as e:
                    print(f"[ERROR] Prediction failed: {e}")
                    predicted_label = ["Unknown"]
                    confidence = 0.0

                print(f"[PREDICTION] {predicted_label[0]} (Confidence: {confidence:.2f})")

                # Update the last detected dance and confidence if confidence is above threshold
                confidence_threshold = 0.5  # You can adjust this threshold
                if confidence >= confidence_threshold:
                    last_detected_dance = predicted_label[0]
                    last_confidence = confidence
                    dance_display_start_time = current_time  # Reset display timer

                last_prediction_time = current_time  # Reset the timer

        # Display the detected dance and confidence
        if last_detected_dance is not None:
            # Check if the dance display duration has not elapsed
            if (current_time - dance_display_start_time) < SIGN_DISPLAY_DURATION:
                # Define the color for the text (Red in BGR format)
                color = (0, 0, 255)  # OpenCV uses BGR, so red is (0, 0, 255)

                # Add a semi-transparent black rectangle as a background for better text visibility
                overlay = frame.copy()
                cv2.rectangle(overlay, (5, 50), (400, 130), (0, 0, 0), -1)  # Black rectangle
                alpha = 0.4  # Transparency factor
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Display the detected dance
                cv2.putText(frame, f"Dance: {last_detected_dance}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2, cv2.LINE_AA)

                # Display the confidence score below the dance name
                cv2.putText(frame, f'Confidence: {last_confidence:.2f}', (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Dance Move Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Exiting...")
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == "__main__":
    main()
    