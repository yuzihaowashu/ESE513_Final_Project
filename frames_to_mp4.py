import cv2
import os
import argparse

def frames_to_mp4(frames_folder, output_video_path, fps=30):
    # Sort images by extracting numeric values from filenames
    images = [img for img in os.listdir(frames_folder) if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    images.sort(key=lambda x: int(''.join(filter(str.isdigit, x))) if any(char.isdigit() for char in x) else x)

    if not images:
        print(f"No images found in {frames_folder}.")
        return

    # Get the width and height of the first image
    first_frame = cv2.imread(os.path.join(frames_folder, images[0]))
    height, width, _ = first_frame.shape
    
    # Define the video writer with codec, fps, and frame size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 'mp4v' for mp4 format
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Write each frame to the video
    for image_name in images:
        frame_path = os.path.join(frames_folder, image_name)
        frame = cv2.imread(frame_path)
        out.write(frame)
    
    # Release the video writer
    out.release()
    print(f"MP4 video saved to '{output_video_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine frames into an MP4 video.")
    parser.add_argument("frames_folder", type=str, help="Directory containing image frames.")
    parser.add_argument("output_video", type=str, help="Output MP4 video file path.")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (optional, default is 30).")

    args = parser.parse_args()

    frames_to_mp4(args.frames_folder, args.output_video, fps=args.fps)

