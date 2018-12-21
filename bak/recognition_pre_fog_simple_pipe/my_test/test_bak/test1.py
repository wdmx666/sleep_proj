"""
@Name:        test1
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/5
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""

import os
import path
from pyhocon import ConfigFactory



# 配置静态数据(变化程度极低的输入)
# dependency inject
class SignalETLConfig:
    from recognition_pre_fog.commons.utils import PreProcedure
    path = r"E:\my_proj\fog_recognition\recognition_pre_fog_pipe\bak\recognition_pre_fog_dag\s1_etl_signal_video\config\para.conf"
    _d_conf = ConfigFactory.parse_file(os.path.abspath(path))
    part_field = _d_conf["SHEET_FIELD"]
    PARTS = _d_conf["PARTS"]
    SHEET_FIELD_MAP = PreProcedure.create_part_full_field_map(_d_conf["SHEET_FIELD"])
    TRANSFORM_MATRIX_PATH = _d_conf["TRANSFORM_MATRIX_PATH"]
    TRANSFORM_MATRIX_FILE = _d_conf["TRANSFORM_MATRIX_FILE"]
    SIGNAL_SAVE_PATH = _d_conf["SIGNAL_SAVE_PATH"]
    RAW_DATA_PATH = r'E:/my_proj/fog_recognition/ExtendFoGData/raw_data'

