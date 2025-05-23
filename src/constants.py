from torchvision import transforms
from transforms import CropBottom


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

ACCELERATION_COLUMNS = \
    [f'PSM{nr}_ee_a_{axis}' for axis in ['x', 'y', 'z'] for nr in [1, 2]] \
    + [f'PSM{nr}_joint_{joint}_a' for joint in range(1, 7) for nr in [1, 2]] \
    + [f'PSM{nr}_jaw_angle_a' for nr in [1, 2]]

TARGET_COLUMNS = ['Force_x_smooth', 'Force_y_smooth', 'Force_z_smooth']

# crop the data to balance the zero and non-zero force data
START_END_TIMES = {
    "force_policy": {
        1: [(450, -1)],
        2: [(800, -1)],
        3: [(800, 1700), (2300, -1)],
        4: [(400, 1000), (2500, -1)],
        5: [(0,-1)],
        6: [(2000, -1)],
        7: [(0, -1)],
        8: [(500, -1)],
        9: [(700, 1700), (2200, -1)],
        10: [(1700, -1)],
        11: [(500, -1)],
        12: [(0, -1)],
        13: [(1100, -1)],
        14: [(1000, -1)],
        15: [(400, -1)],
        16: [(200, -1)],
        17: [(200, 600)],
        18: [(100, 450), (550, 700), (800, -1)],
        19: [(100, 300), (550, 800)],
        20: [(150, 300)],
        21: [(200, 500), (850, -1)],
        22: [(300, 600)],
        23: [(100, 350), (450, 700), (800, -1)],
        24: [(200, -1)],
        25: [(200, -1)],
        26: [(200, 550)],
        27: [(200, -1)],
        28: [(400, -1)],
        29: [(200, -1)],
        30: [(200, 500)],
        31: [(200, -1)],
        32: [(150, -1)],
        33: [(250, -1)],
        34: [(250, -1)],
        35: [(0, -1)],
        36: [(350, -1)],
        37: [(350, -1)],
        38: [(500, -1)],
        39: [(400, -1)],
        40: [(400, -1)],
        41: [(200, -1)],
        42: [(0, -1)],
        43: [(450, -1)],
        44: [(400, -1)],
        45: [(400, -1)],
        46: [(300, -1)],
        47: [(350, -1)],
        48: [(150, -1)],
        49: [(0, 400)],
        50: [(200, -1)]
    },
    "no_force_policy": {
        1: [(500, -1)],
        3: [(700, -1)],
        4: [(500, -1)]
    }
}

# for runs that have more than one peak, we need to specify the time ranges 
# for each peak for use in the evaluation script
INDIVIDUAL_PEAK_TIMES = {
    "force_policy": {
        1: [(0, -1)],
        2: [(0, -1)],
        3: [(0, 2000), (2001, -1)],
        4: [(0, 2000), (2001, -1)],
        5: [(0,-1)],
        6: [(0, -1)],
        7: [(0, -1)],
        8: [(0, -1)],
        9: [(0, 2000), (2001, -1)],
        10: [(0, -1)],
        11: [(0, -1)],
        12: [(0, -1)],
        13: [(0, -1)],
        14: [(0, -1)],
        15: [(0, -1)],
        16: [(0, -1)],
        17: [(0, -1)],
        18: [(0, 450), (451, 750), (751, -1)],
        19: [(0, -1)],
        20: [(0, -1)],
        21: [(0, 500), (500, -1)],
        22: [(0, -1)],
        23: [(0, 400), (401, 700), (701, -1)],
        24: [(0, -1)],
        25: [(0, -1)],
        26: [(0, -1)],
        27: [(0, -1)],
        28: [(0, -1)],
        29: [(0, -1)],
        30: [(0, -1)],
        31: [(0, -1)],
        32: [(0, -1)],
        33: [(0, -1)],
        34: [(0, -1)],
        35: [(0, -1)],
        36: [(0, -1)],
        37: [(0, -1)],
        38: [(0, 600), (601, -1)],
        39: [(0, -1)],
        40: [(0, -1)],
        41: [(0, -1)],
        42: [(0, -1)],
        43: [(0, -1)],
        44: [(0, -1)],
        45: [(0, -1)],
        46: [(0, -1)],
        47: [(0, -1)],
        48: [(0, -1)],
        49: [(0, -1)],
        50: [(0, -1)]
    }
}
EXCEL_FILE_NAMES = {
    "force_policy": {
        key: (f"dec6_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 1 <= key <= 15 else
              f"dec19_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 16 <= key <= 30 else
              f"dec20_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx")
        for key in range(1, 51)
    },
    "no_force_policy": {
        key: (f"dec6_no_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 1 <= key <= 15 else
              f"dec19_no_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx" if 16 <= key <= 30 else
              f"dec20_no_force_no_TA_lastP_randomPosHeight_cs100_run{key}.xlsx")
        for key in range(1, 51)
    },
}


NUM_IMAGE_FEATURES = 30
NUM_ROBOT_FEATURES = 58
NUM_ROBOT_FEATURES_INCL_ACCEL = 78
CNN_MODEL_VERSION = "efficientnet_v2_m"
FREQUENCY = 800 # Hz


# Transformer Config
SEQ_LENGTH = 10
HIDDEN_LAYERS = [128, 256]
NUM_HEADS = 4
NUM_ENCODER_LAYERS = 4
NUM_DECODER_LAYERS = 2
DIM_FEEDFORWARD = 256
DROPOUT_RATE = 0.3

DEFAULT_TEST_RUNS = [[13, 29, 33, 36, 39, 45], []]

RES_NET_TRAIN_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

RES_NET_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    CropBottom((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

ENCODER_WEIGHTS_FN = "encoder_weights.pth"
TARGET_SCALER_FN = "transformations/target_scaler.joblib"
FEATURE_SCALER_FN = "transformations/feature_scaler.joblib"

MOVING_AVG_WINDOW_SIZE = 5
