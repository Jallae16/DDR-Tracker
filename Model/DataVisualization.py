import pandas as pd
import numpy as np
import cv2
import math

def load_csv(csv_path):
    """
    Load the CSV file into a pandas DataFrame.
    """
    print(f"[INFO] Loading CSV file from {csv_path}...")
    df = pd.read_csv(csv_path)
    print(f"[INFO] CSV file loaded. Total samples: {df['sample_number'].nunique()}")
    return df

def get_landmark_columns():
    """
    Generate a list of landmark columns based on the provided format.
    """
    landmarks = []
    for i in range(33):  # MediaPipe Pose has 33 landmarks
        for coord in ['x', 'y', 'z', 'visibility']:
            landmarks.append(f'landmark_{i}_{coord}')
    return landmarks

def get_pose_connections():
    """
    Define the connections between landmarks as per MediaPipe's Pose specification.
    This list defines which landmarks should be connected to each other.
    """
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
    """
    Calculate the number of rows and columns for the grid layout based on the number of samples.
    """
    columns = min(num_samples, max_columns)
    rows = math.ceil(num_samples / columns) if columns else 1
    return rows, columns

def visualize_all_dances(df, fps=10, canvas_size=(320, 240), max_columns=5, samples_per_page=10):
    """
    Visualize dance samples in a paginated grid layout.

    Args:
        df (pd.DataFrame): DataFrame containing the dance data.
        fps (int): Frames per second for the visualization.
        canvas_size (tuple): Size of each individual sample's visualization window (width, height).
        max_columns (int): Maximum number of columns in the grid.
        samples_per_page (int): Number of samples to display per page.
    """
    sample_numbers = sorted(df['sample_number'].unique())
    num_samples = len(sample_numbers)
    print(f"[INFO] Number of samples to visualize: {num_samples}")

    landmark_cols = get_landmark_columns()
    connections = get_pose_connections()

    # Calculate total number of pages
    total_pages = math.ceil(num_samples / samples_per_page)
    current_page = 0  # Zero-based indexing

    print(f"[INFO] Total pages: {total_pages}")

    # Prepare sample data and initialize frame indices
    samples_data = {
        num: df[df['sample_number'] == num].sort_values(by='frame_number').reset_index(drop=True)
        for num in sample_numbers
    }
    frame_indices = {num: 0 for num in sample_numbers}

    # Set up canvas dimensions
    width, height = canvas_size
    # Define grid layout per page
    rows_per_page, columns_per_page = calculate_grid(samples_per_page, max_columns=max_columns)
    total_width = width * columns_per_page
    total_height = height * rows_per_page

    # Initialize OpenCV window
    window_name = "Dance Visualization - All Samples (Press 'x' for Next, 'z' for Previous, 'q' to Quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, total_width, total_height)

    print("[INFO] Starting visualization. Press 'x' for next page, 'z' for previous page, 'q' to quit.")

    while True:
        # Calculate the range of samples for the current page
        start_idx = current_page * samples_per_page
        end_idx = min(start_idx + samples_per_page, num_samples)
        current_page_samples = sample_numbers[start_idx:end_idx]
        current_page_count = len(current_page_samples)

        # Calculate grid layout for the current page
        rows, columns = calculate_grid(current_page_count, max_columns=max_columns)

        # Adjust total canvas size for the current page
        total_canvas_width = width * columns
        total_canvas_height = height * rows
        total_canvas = np.zeros((total_canvas_height, total_canvas_width, 3), dtype=np.uint8)

        for idx, sample_num in enumerate(current_page_samples):
            sample_df = samples_data[sample_num]
            frame_idx = frame_indices[sample_num]

            if frame_idx >= len(sample_df):
                frame_idx = 0  # Loop back to the first frame
                frame_indices[sample_num] = 0

            row = sample_df.iloc[frame_idx]
            landmarks = row[landmark_cols].values.reshape(-1, 4)  # Shape: (33, 4)

            # Scale landmarks to fit the canvas
            landmark_points = []
            for lm in landmarks:
                x_norm, y_norm, z, visibility = lm
                x = int(x_norm * width)
                y = int(y_norm * height)
                landmark_points.append((x, y))

            # Create a blank canvas for the sample
            sample_canvas = np.zeros((height, width, 3), dtype=np.uint8)

            # Draw connections
            for connection in connections:
                start_idx_conn, end_idx_conn = connection
                start_point = landmark_points[start_idx_conn]
                end_point = landmark_points[end_idx_conn]
                # Check if visibility is above a threshold (e.g., 0.5) for both landmarks
                if landmarks[start_idx_conn][3] > 0.5 and landmarks[end_idx_conn][3] > 0.5:
                    cv2.line(sample_canvas, start_point, end_point, (0, 0, 255), 2)

            # Draw landmarks
            for point, lm in zip(landmark_points, landmarks):
                x, y = point
                visibility = lm[3]
                if visibility > 0.5:
                    cv2.circle(sample_canvas, (x, y), 5, (0, 255, 0), -1)

            # Add labels for dance name and frame number
            cv2.putText(sample_canvas, f"Dance: {row['dance_name']}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(sample_canvas, f"Frame: {row['frame_number']}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Calculate the position on the total canvas
            row_idx = idx // columns
            col_idx = idx % columns
            x_offset = col_idx * width
            y_offset = row_idx * height

            # Place the sample canvas onto the total canvas
            total_canvas[y_offset:y_offset + height, x_offset:x_offset + width] = sample_canvas

            # Update the frame index for the sample
            frame_indices[sample_num] += 1

        # Overlay the page number on the total canvas
        cv2.putText(total_canvas, f"Page: {current_page + 1}/{total_pages}", 
                    (10, total_canvas_height - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        # Display the total canvas
        cv2.imshow(window_name, total_canvas)

        # Wait for keypress
        key = cv2.waitKey(int(1000 / fps)) & 0xFF

        if key == ord('q'):
            print("[INFO] Visualization terminated by user.")
            break
        elif key == ord('x'):
            # Move to the next page
            if current_page < total_pages - 1:
                current_page += 1
                print(f"[INFO] Moved to page {current_page + 1}/{total_pages}.")
            else:
                print("[INFO] Reached the last page. Looping back to the first page.")
                current_page = 0
        elif key == ord('z'):
            # Move to the previous page
            if current_page > 0:
                current_page -= 1
                print(f"[INFO] Moved to page {current_page + 1}/{total_pages}.")
            else:
                print("[INFO] Already at the first page. Moving to the last page.")
                current_page = total_pages - 1

    # Close all OpenCV windows upon exit
    cv2.destroyAllWindows()

def main():
    csv_path = 'dance_dataset/all_samples.csv'  # Update this path if necessary
    df = load_csv(csv_path)
    visualize_all_dances(df, fps=10, canvas_size=(320, 240), max_columns=5, samples_per_page=10)

if __name__ == "__main__":
    main()
