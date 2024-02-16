import os
from typing import List, Optional, Tuple
import pandas as pd
import numpy as np

FEATURE_COLUMS = ['PSM1_joint_1', 'PSM1_joint_2', 'PSM1_joint_3', 'PSM1_joint_4',
                  'PSM1_joint_5', 'PSM1_joint_6', 'PSM1_jaw_angle', 'PSM1_ee_x',
                  'PSM1_ee_y', 'PSM1_ee_z', 'PSM1_Orientation_Matrix_[1,1]',
                  'PSM1_Orientation_Matrix_[1,2]', 'PSM1_Orientation_Matrix_[1,3]',
                  'PSM1_Orientation_Matrix_[2,1]', 'PSM1_Orientation_Matrix_[2,2]',
                  'PSM1_Orientation_Matrix_[2,3]', 'PSM1_Orientation_Matrix_[3,1]',
                  'PSM1_Orientation_Matrix_[3,2]', 'PSM1_Orientation_Matrix_[3,3]',
                  'PSM2_joint_1', 'PSM2_joint_2', 'PSM2_joint_3', 'PSM2_joint_4',
                  'PSM2_joint_5', 'PSM2_joint_6', 'PSM2_jaw_angle', 'PSM2_ee_x',
                  'PSM2_ee_y', 'PSM2_ee_z', 'PSM2_Orientation_Matrix_[1,1]',
                  'PSM2_Orientation_Matrix_[1,2]', 'PSM2_Orientation_Matrix_[1,3]',
                  'PSM2_Orientation_Matrix_[2,1]', 'PSM2_Orientation_Matrix_[2,2]',
                  'PSM2_Orientation_Matrix_[2,3]', 'PSM2_Orientation_Matrix_[3,1]',
                  'PSM2_Orientation_Matrix_[3,2]', 'PSM2_Orientation_Matrix_[3,3]']

IMAGE_COLUMS = ['ZED Camera Left', 'ZED Camera Right']

TARGET_COLUMNS = ['Force_x_smooth', 'Force_y_smooth', 'Force_z_smooth']


def get_img_paths(cam: str, excel_df: pd.DataFrame) -> List[str]:
    assert cam in ["Left", "Right"], f"Invalid {cam}"

    col_name = f"ZED Camera {cam}"
    img_paths = []
    for server_path in excel_df[col_name].to_list():
        dirs = server_path.split("/")
        new_path = "/".join(dirs[-3:])
        img_paths.append(new_path)

    return img_paths


def load_data(excel_file_names: List[str]) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    assert isinstance(excel_file_names, list)

    all_X = []
    all_y = []
    all_img_left_paths = []
    all_img_right_paths = []

    for excel_file_name in excel_file_names:
        print(f"Loading data: {excel_file_name}")
        relevant_cols = FEATURE_COLUMS + IMAGE_COLUMS + TARGET_COLUMNS
        excel_df = pd.read_excel(excel_file_name, usecols=relevant_cols)

        X = excel_df[FEATURE_COLUMS].to_numpy()
        y = excel_df[TARGET_COLUMNS].to_numpy()
        img_left_paths = get_img_paths("Left", excel_df)
        img_right_paths = get_img_paths("Right", excel_df)

        all_X.append(X)
        all_y.append(y)
        all_img_left_paths += img_left_paths
        all_img_right_paths += img_right_paths

    all_X = np.concatenate(all_X, axis=0)
    all_y = np.concatenate(all_y, axis=0)

    return all_X, all_y, all_img_left_paths, all_img_right_paths


def load_dataset(path: str, run_nums: Optional[List[int]] = None):
    assert os.path.isdir(path), f"{path} is not a directory"
    assert os.path.exists(os.path.join(path, "images")), \
        f"{path} does not contain an images directory"
    assert os.path.exists(os.path.join(path, "roll_out")), \
        f"{path} does not contain a roll out directory"

    roll_out_dir = os.path.join(path, "roll_out")

    if run_nums is None:
        excel_files = [os.path.join(roll_out_dir, f)
                       for f in os.listdir(roll_out_dir)]
    else:
        assert isinstance(run_nums, list)
        assert len(run_nums) > 0
        excel_files = [
            f"{roll_out_dir}/dec6_force_no_TA_lastP_randomPosHeight_cs100_run{n}.xlsx" for n in run_nums]

    return load_data(excel_files)


if __name__ == "__main__":
    all_X, all_y, all_img_left_paths, all_img_right_paths = load_dataset(
        path="data/train", run_nums=[1, 2])

    assert all_X.shape == (2549, 38)
    assert all_y.shape == (2549, 3)

    assert len(all_img_left_paths) == 2549
    assert len(all_img_right_paths) == 2549
