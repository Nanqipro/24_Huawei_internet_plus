import cv2
import os


def extract_frames(video_path, output_folder):
    # Extract the video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Create a subfolder in the output folder for the current video
    video_output_folder = os.path.join(output_folder, video_name)
    if not os.path.exists(video_output_folder):
        os.makedirs(video_output_folder)
        print(f"Created directory: {video_output_folder}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return

    frame_count = 0
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame was read successfully, ret is True
        if not ret:
            break

        # Save the frame as an image file
        frame_filename = os.path.join(video_output_folder, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_filename, frame)

        # Print progress
        print(f"Saved {frame_filename}")

        frame_count += 1

    # Release the video capture object
    cap.release()
    cv2.destroyAllWindows()

    print(f"Extracted {frame_count} frames from {video_path} to {video_output_folder}")


def extract_frames_from_videos(input_folder, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created directory: {output_folder}")

    # Get list of all video files in the input folder
    video_files = [f for f in os.listdir(input_folder) if os.path.isfile(os.path.join(input_folder, f)) and f.endswith(
        ('.mp4', '.avi'))]  # Add video extensions filter

    print(f"Found {len(video_files)} video files in {input_folder}")

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)

        print(f"Processing {video_path}...")
        extract_frames(video_path, output_folder)  # Pass the same output_folder for all videos


# Example usage
input_folder = r'C:\Users\nanqipro\Desktop\test\疲劳驾驶检测视频数据\video2044\test'  # Replace with your input folder path
output_folder = r'C:\Users\nanqipro\Desktop\test\疲劳驾驶检测视频数据\video2044\res'  # Replace with your output folder path
extract_frames_from_videos(input_folder, output_folder)
