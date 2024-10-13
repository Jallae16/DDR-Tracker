import cv2
import mediapipe as mp
import numpy as np
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
    model = load_model(model_path)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(label_encoder_path)
    max_seq_length = joblib.load(max_seq_length_path)
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

    # Start capturing
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
            # Optionally, visualize landmarks on the frame
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
            )
        else:
            pass

        # Check if it's time to make a prediction
        current_time = time.time()
        if (current_time - last_prediction_time) >= prediction_interval:
            if len(frame_buffer) == 0:
                print("[WARNING] No frames to make a prediction.")
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
                sequence_scaled = scaler.transform(sequence)

                # Pad sequences if necessary (already handled above)
                # Compute variance features
                variance = np.var(sequence_scaled, axis=0).reshape(1, -1)

                # Expand dimensions for batch size
                sequence_scaled = np.expand_dims(sequence_scaled, axis=0)  # Shape: (1, max_seq_length, features)

                # Make prediction
                prediction = model.predict([sequence_scaled, variance])
                predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
                confidence = np.max(prediction)

                print(f"[PREDICTION] {predicted_label[0]} (Confidence: {confidence:.2f})")

                cv2.putText(frame, f"Dance Move: {predicted_label[0]} ({confidence*100:.1f}%)",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            last_prediction_time = current_time  # Reset the timer

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
