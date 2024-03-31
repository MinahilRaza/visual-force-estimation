from PIL import Image
import os
import argparse

NEW_SIZE = 256


def parse_cmd_line() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively crops every .png image provided in the directory")
    parser.add_argument("-d", "--directory", required=True)
    return parser.parse_args()


def resize_and_crop_images(input_directory: str):
    if not os.path.isdir(input_directory):
        print("Directory does not exist:", input_directory)
        return

    print(f"Processing: {input_directory}")

    for filename in os.listdir(input_directory):
        if os.path.isdir(os.path.join(input_directory, filename)):
            resize_and_crop_images(
                input_directory=os.path.join(input_directory, filename))
        elif filename.endswith('.png'):
            file_path = os.path.join(input_directory, filename)
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

            except Exception as e:
                print(f"Failed to process {filename}: {e}")


if __name__ == "__main__":
    args = parse_cmd_line()
    resize_and_crop_images(args.directory)
