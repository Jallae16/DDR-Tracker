import pandas as pd
import numpy as np
import cv2

def load_csv(csv_path):
    print(f"[INFO] Loading CSV file from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV file loaded. Total samples: {df['sample_number'].nunique()}")
    return df

def get_landmark_columns():
    landmarks = []
    for i in range(33):  # MediaPipe Pose has 33 landmarks
        for coord in ['x', 'y', 'z', 'visibility']:
            landmarks.append(f'landmark_{i}_{coord}')
    return landmarks

def get_pose_connections():
    return [
        (0, 1), (1, 2), (2, 3), (3, 7),
        (0, 4), (4, 5), (5, 6), (6, 8),
        (9, 10), (11, 12), (11, 13), (13, 15),
        (12, 14), (14, 16), (15, 17), (16, 18),
        (5, 19), (6, 20), (5, 11), (6, 12),
        (11, 23), (12, 24), (23, 24), (23, 25),
        (24, 26), (25, 27), (26, 28), (27, 29),
        (28, 30), (29, 31), (30, 32), (31, 32)
    ]

def calculate_grid(num_samples, max_columns=5):
    columns = min(num_samples, max_columns)
    rows = (num_samples + columns - 1) // columns  # Ceiling division
    return rows, columns

def visualize_all_dances(df, fps=30, canvas_size=(320, 240), max_columns=5):
    sample_numbers = sorted(df['sample_number'].unique())
    num_samples = len(sample_numbers)
    print(f"[INFO] Number of samples to visualize: {num_samples}")

    landmark_cols = get_landmark_columns()
    connections = get_pose_connections()

    # Calculate the grid layout
    rows, columns = calculate_grid(num_samples, max_columns)
    print(f"[INFO] Grid layout: {rows}x{columns}")

    # Prepare sample data and frame indices
    samples_data = {num: df[df['sample_number'] == num].sort_values(by='frame_number').reset_index(drop=True)
                    for num in sample_numbers}
    frame_indices = {num: 0 for num in sample_numbers}

    # Set up canvas dimensions
    width, height = canvas_size
    total_width = width * columns
    total_height = height * rows

    # Initialize OpenCV window
    window_name = "Dance Visualization - All Samples"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, total_width, min(total_height, 1080))  # Cap to screen height for scrolling

    print("[INFO] Starting visualization. Press 'q' to quit.")

    while True:
        total_canvas = np.zeros((total_height, total_width, 3), dtype=np.uint8)

        for idx, sample_num in enumerate(sample_numbers):
            sample_df = samples_data[sample_num]
            frame_idx = frame_indices[sample_num]

            if frame_idx >= len(sample_df):
                frame_idx = 0  # Loop back
                frame_indices[sample_num] = 0

            row = sample_df.iloc[frame_idx]
            landmarks = row[landmark_cols].values.reshape(-1, 4)

            landmark_points = [(int(lm[0] * width), int(lm[1] * height)) for lm in landmarks]

            sample_canvas = np.zeros((height, width, 3), dtype=np.uint8)
            for connection in connections:
                cv2.line(sample_canvas, landmark_points[connection[0]], 
                         landmark_points[connection[1]], (0, 0, 255), 2)

            for point in landmark_points:
                cv2.circle(sample_canvas, point, 5, (0, 255, 0), -1)

            # Add labels for dance name and frame number
            cv2.putText(sample_canvas, f"Dance: {row['dance_name']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(sample_canvas, f"Frame: {row['frame_number']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Calculate the canvas position
            row_idx = idx // columns
            col_idx = idx % columns
            x_offset = col_idx * width
            y_offset = row_idx * height

            total_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = sample_canvas

            frame_indices[sample_num] += 1

        # Display the canvas
        cv2.imshow(window_name, total_canvas)

        key = cv2.waitKey(int(1000 / fps)) & 0xFF
        if key == ord('q'):
            print("[INFO] Visualization terminated by user.")
            break

    cv2.destroyAllWindows()

def main():
    csv_path = 'dance_dataset/all_samples.csv'  # Update this path if necessary
    df = load_csv(csv_path)
    visualize_all_dances(df, fps=30, canvas_size=(320, 240), max_columns=5)

if __name__ == "__main__":
    main()
