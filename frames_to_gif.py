import os
import sys
import imageio.v2 as imageio  # Import imageio.v2 to maintain compatibility
import argparse

def frames_to_gif(frames_folder, output_gif_path, fps=10):
    images = [img for img in os.listdir(frames_folder) if img.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))]
    images.sort()

    if not images:
        print(f"No images found in {frames_folder}.")
        return

    frames = []
    for image_name in images:
        frame_path = os.path.join(frames_folder, image_name)
        frame = imageio.imread(frame_path)
        frames.append(frame)

    duration = 1000 / fps 

    # Save frames as a GIF
    imageio.mimsave(output_gif_path, frames, duration=duration)
    print(f"GIF saved to '{output_gif_path}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Combine frames into a GIF.')
    parser.add_argument('frames_folder', type=str, help='Directory containing image frames.')
    parser.add_argument('output_gif', type=str, help='Output GIF file path.')
    parser.add_argument('fps', type=float, nargs='?', default=10, help='Frames per second (optional, default is 10).')

    args = parser.parse_args()

    frames_to_gif(args.frames_folder, args.output_gif, fps=args.fps)


    #how to run in command line
    #ex: python frames_to_gif.py frames_folder output_gif [fps]
    #frames_folder: directory that stores all the frames
    #output_gif: the name that you want for your gif
    #fps: set the fps of the video

