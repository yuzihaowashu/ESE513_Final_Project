import cv2
import os
import argparse

def extract_frames(output_folder, video_path):
    os.makedirs(output_folder, exist_ok=True)
    
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    frame_count = 0
    
    while success:
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, image)
        success, image = vidcap.read()
        frame_count += 1
    
    vidcap.release()
    print(f"Extracted {frame_count} frames to '{output_folder}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract frames from a video.')
    parser.add_argument('output_folder', type=str, help='Directory to save extracted frames.')
    parser.add_argument('video_file', type=str, help='Path to the input video file.')
    args = parser.parse_args()
    
    extract_frames(args.output_folder, args.video_file)

    #how to run in command line
    #python extract_frames.py output_folder video_file
    #output_folder: the directory of where to store all the frames
    #video_file: video that is going to be extracted
