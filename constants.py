LAYOUT = {
    "Training Plots": {
        "MSE": ["Multiline", ["MSE/train", "MSE/test"]],
        "RMSE": ["Multiline", ["RMSE/train", "RMSE/test"]],
    },
}

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

TIME_COLUMN = ["Time (Seconds)"]

VELOCITY_COLUMNS = \
    [f'PSM{nr}_ee_v_{axis}' for axis in ['x', 'y', 'z'] for nr in [1, 2]] \
    + [f'PSM{nr}_joint_{joint}_v' for joint in range(1, 7) for nr in [1, 2]] \
    + [f'PSM{nr}_jaw_angle_v' for nr in [1, 2]]

TARGET_COLUMNS = ['Force_x_smooth', 'Force_y_smooth', 'Force_z_smooth']

START_END_TIMES = {
    "force_policy": {
        1: [(400, -1)],
        2: [(700, -1)],
        3: [(800, -1)],
        4: [(400, 1200), (2500, -1)],
        6: [(2000, -1)],
        8: [(500, -1)],
        9: [(700, 1700), (2200, -1)],
        10: [(1700, -1)],
        11: [(500, -1)]
    },
    "no_force_policy": {
        1: [(500, -1)],
        4: [(500, -1)]
    }
}

EXCEL_FILE_NAMES = {
    "force_policy": {
        1: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run1.xlsx",
        2: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run2.xlsx",
        3: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run3.xlsx",
        4: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run4.xlsx",
        6: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run6.xlsx",
        7: "corrupted_file",
        8: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run8.xlsx",
        9: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run9.xlsx",
        10: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run10.xlsx",
        11: "dec6_force_no_TA_lastP_randomPosHeight_cs100_run11.xlsx"
    },
    "no_force_policy": {
        1: "dec6_no_force_no_TA_lastP_randomPosHeight_cs100_run1.xlsx",
        4: "dec6_no_force_no_TA_lastP_randomPosHeight_cs100_run4.xlsx",
    }
}

NUM_IMAGE_FEATURES = 30
NUM_ROBOT_FEATURES = 58
