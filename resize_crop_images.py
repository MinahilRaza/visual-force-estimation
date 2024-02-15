from PIL import Image
import os

NEW_SIZE = 256


def resize_and_crop_images(input_directory: str):
    if not os.path.isdir(input_directory):
        print("Directory does not exist:", input_directory)
        return

    print(f"Processing: {input_directory}")

    for filename in os.listdir(input_directory):
        if filename.endswith('.png'):
            file_path = os.path.join(input_directory, filename)
            new_img_file_path = f"{input_directory}/processed/{filename}"
            try:
                with Image.open(file_path) as img:
                    width, height = img.size
                    new_height = NEW_SIZE
                    new_width = NEW_SIZE

                    left = (width - new_width) // 2
                    top = height - new_height
                    right = (width + new_width) // 2
                    bottom = height

                    img_cropped = img.crop((left, top, right, bottom))
                    img_cropped.save(file_path)

                    print(f"Processed {filename}")

            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    image_path = "data/train/images"
    image_directories = [os.path.join(image_path, f)
                         for f in os.listdir(image_path)]
    for roll_out_images in image_directories:
        resize_and_crop_images(roll_out_images)
