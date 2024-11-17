import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw
def overlay_zoomed_section_fixed(image_path, top_left_corner, box_size=15, zoom_factor=2, overlay_position="top_right"):
    img = Image.open(image_path)
    width, height = img.size

    x, y = top_left_corner
    zoom_box = (x, y, x + box_size, y + box_size)

    cropped = img.crop(zoom_box)

    zoomed = cropped.resize((box_size * zoom_factor, box_size * zoom_factor), Image.Resampling.LANCZOS)
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)

    x, y = top_left_corner
    zoom_box = (x, y, x + box_size, y + box_size)

    # Outline the zoom box in red
    draw.rectangle(zoom_box, outline="red", width=1)

    overlay_x, overlay_y = {
        "top_right": (width - zoomed.width, 0),
        "top_left": (0, 0),
        "bottom_right": (width - zoomed.width, height - zoomed.height),
        "bottom_left": (0, height - zoomed.height)
    }[overlay_position]

    img.paste(zoomed, (overlay_x, overlay_y))
    overlay_box = (overlay_x, overlay_y, overlay_x + zoomed.width, overlay_y + zoomed.height)
    draw.rectangle(overlay_box, outline="green", width=1)


    return img

def process_and_display_images_fixed(input_dir, top_left_corner, box_size=15, zoom_factor=2, overlay_position="top_right"):
    images = [
        os.path.join(input_dir, filename)
        for filename in os.listdir(input_dir)
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    if not images:
        print("No valid images found in the directory.")
        return

    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))

    if len(images) == 1:
        axes = [axes]  

    for ax, image_path in zip(axes, images):
        try:
            processed_image = overlay_zoomed_section_fixed(image_path, top_left_corner, box_size, zoom_factor, overlay_position)

            ax.imshow(processed_image)
            ax.axis('off')
            ax.set_title(os.path.basename(image_path))
        except Exception as e:
            print(f"Failed to process {image_path}: {e}")

    plt.tight_layout()
    plt.show()

#example usage
input_directory = "test"  #replace with the actual directory containing images
top_left_corner = (50, 50)  #adjust the top-left corner of the zoomed-in area

process_and_display_images_fixed(input_directory, top_left_corner, box_size=15, zoom_factor=3, overlay_position="top_left")

#you need to call
#process_and_display_images_fixed(input_dir, top_left_corner, box_size=15, zoom_factor=2, overlay_position="top_right")
#input_dir: directory with images stored
#top_left_corener: cordinage of the top left corner of the zoomed box
#box_size: size of the box (in pixels)
#zoom_factor: how much you want to zoome in
#overlay_position: where you want your overlay to be
