import os
import shutil
import random
from typing import List

source_dir = './data'

train_dir = './data/train/roll_out'
test_dir = './data/test/roll_out'
val_dir = './data/val/roll_out'


def move_files(files: List[str], source: str, destination: str):
    for file in files:
        shutil.move(os.path.join(source, file),
                    os.path.join(destination, file))


def get_files_in_dir(input_dir: str) -> List[str]:
    return [f for f in os.listdir(input_dir) if f.endswith(
        '.xlsx') and os.path.isfile(os.path.join(input_dir, f))]


for directory in [train_dir, test_dir, val_dir]:
    if os.path.exists(directory):
        old_files = get_files_in_dir(directory)
        move_files(old_files, directory, source_dir)
    else:
        os.makedirs(directory)

files = get_files_in_dir(source_dir)

random.shuffle(files)

train_files = files[:70]
test_files = files[70:90]
val_files = files[90:]


move_files(train_files, source_dir, train_dir)
move_files(test_files, source_dir, test_dir)
move_files(val_files, source_dir, val_dir)

print("Files have been successfully distributed into train, test, and val directories.")
