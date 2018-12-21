
from enum import Enum
from joblib import Memory
location = './cachedir'
memory = Memory(location, verbose=0)


class MyNode(Enum):
    SignalETLInit = "SignalETLInit"
    SignalETL = "SignalETL"
    VideoMarkETL = "VideoMarkETL"
    SignalMark = "SignalMark"
    RemarkScaleFeature = "RemarkScaleFeature"
    ScaleFeatureOneByOne = "ScaleFeatureOneByOne"
    ValidationProcessor = "ValidationProcessor"

    VideoUnfold4FoG = "VideoUnfold4FoG"
    SignalReMark4FoG = "SignalReMark4FoG"
    ScaleFeatureSelect = "ScaleFeatureSelect123"
    FeatureReETL = "FeatureReETL"
    DOE2 = "ONE_ALL"






COLS_NAME = {"signal_cols":
    ["A100_accelerometer_x","A100_accelerometer_y","A100_accelerometer_z","A100_gyroscope_x","A100_gyroscope_y","A100_gyroscope_z",
     "A200_accelerometer_x","A200_accelerometer_y","A200_accelerometer_z","A200_gyroscope_x","A200_gyroscope_y","A200_gyroscope_z",
     "A300_accelerometer_x","A300_accelerometer_y","A300_accelerometer_z","A300_gyroscope_x","A300_gyroscope_y","A300_gyroscope_z",
     "B100_accelerometer_x","B100_accelerometer_y","B100_accelerometer_z","B100_gyroscope_x","B100_gyroscope_y","B100_gyroscope_z",
     "B200_accelerometer_x","B200_accelerometer_y","B200_accelerometer_z","B200_gyroscope_x","B200_gyroscope_y","B200_gyroscope_z",
     "C100_accelerometer_x","C100_accelerometer_y","C100_accelerometer_z","C100_gyroscope_x","C100_gyroscope_y","C100_gyroscope_z",
     "C200_accelerometer_x","C200_accelerometer_y","C200_accelerometer_z","C200_gyroscope_x","C200_gyroscope_y","C200_gyroscope_z"]
     }