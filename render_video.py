import imageio.v2 as imageio
import os
from natsort import natsorted
import argparse

def create_video_from_images(image_folder, output_video, fps=30):
    # Collect and sort images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg'))]
    image_files = natsorted(image_files)

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_video), exist_ok=True)

    # Write video
    writer = imageio.get_writer(output_video, fps=fps)

    for filename in image_files:
        filepath = os.path.join(image_folder, filename)
        image = imageio.imread(filepath)
        writer.append_data(image)

    writer.close()
    print(f"Video saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert a folder of images into a video.")
    parser.add_argument("--image_folder", type=str, help="Path to the folder containing images")
    parser.add_argument("--output", type=str, default="render_video.mp4", help="Output video filename (default: render_video.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    
    args = parser.parse_args()

    create_video_from_images(args.image_folder, args.output, args.fps)
