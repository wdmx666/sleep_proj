# ---------------------------------------------------------
# Name:        configuration
# Description: separate the logic and the create of object, we use a two layer(config and processor),
# if not use this thought we will get a lot of class just creation
# Author:      Lucas Yu
# Created:     2018-04-06
# Copyright:   (c) Zhenluo,Shenzhen,Guangdong 2018
# Licence:
# ---------------------------------------------------------

import os,path
from pyhocon import ConfigFactory
from ..client.proj_config import MyNode
from ..commons.common import MyProperties, MyConfig


class SignalRemarkExtractConfig(MyConfig):
    path = '../recognition_pre_fog_dag/s3_signal_remark_extract/config/para.conf'
    _d_conf = ConfigFactory.parse_file(os.path.abspath(path))

    WINDOW_PARAMETER = _d_conf["WINDOW_PARAMETER"]
    PARALLEL_SIZE = int(_d_conf["PARALLEL_SIZE"])
    STATUS_DEFINITION = _d_conf["STATUS_DEFINITION"]
    PRE_FOG_TIME_LEN = _d_conf['PRE_FOG_TIME_LEN']
    SIGNAL_REMARK_PATH = _d_conf["SIGNAL_REMARK_PATH"]

    STATUS_INFO_COLS = _d_conf["STATUS_INFO_COLS"]
    STATUS_VALUES = _d_conf["STATUS_VALUES"]
    FEATURE_VIDEO_PATH = _d_conf["FEATURE_VIDEO_PATH"]

    INFO_COLS = ["time10", "label_start_time", "label_end_time", "gait_type",
                 "label_confidence", "stimulate_type", "start_time", "end_time", "unit_confidence"]

    COLS_NAME = _d_conf["COLS_NAME"]

    FEATURES = ["F01", "F02", "F03", "F04", "F05", "F06", "F07", "F08", "F09", "F10", "F11",
                "F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19", "F20", "F21", "F22"]

    # "F23", "F24", "F25","F26", "F27", "F28"]
    @classmethod
    def remarkScaleFeatureService(cls):
        from .services import StatusReMarkService, ScaleService, FeatureService, RemarkScaleFeatureService
        rsfs = RemarkScaleFeatureService()
        sr = StatusReMarkService()
        sr.set_para_with_prop({"status_definition": cls.STATUS_DEFINITION, "pre_fog_time_len":cls.PRE_FOG_TIME_LEN})
        ss = ScaleService()
        ss.set_para_with_prop({'signal_cols': cls.COLS_NAME['signal_cols']})
        fs = FeatureService()
        fs.set_para_with_prop({'signal_cols': cls.COLS_NAME['signal_cols'],
                               'features': cls.FEATURES, 'window': cls.calWindow()})
        cls.check_object(sr)
        cls.check_object(ss)
        cls.check_object(fs)
        rsfs.set_para_with_prop({'StatusReMarkService': sr, 'ScaleService': ss, 'FeatureService': fs})
        return rsfs

    @classmethod
    def remarkScaleFeature(cls, name=MyNode.RemarkScaleFeature.name, dependencies=MyNode.SignalMark.name, reset=False):
        from .controllers import RemarkScaleFeature
        processor = RemarkScaleFeature(name, dependencies, reset)
        print('-------------> ', processor.para['RemarkScaleFeatureService'])
        processor.set_para_with_prop({'RemarkScaleFeatureService': cls.remarkScaleFeatureService()})
        print('-------------> ', processor.para['RemarkScaleFeatureService'])
        return processor

    @classmethod
    def calWindow(cls):
        from .algo_general import CalWindow
        wd = CalWindow(start=0, ksize=256, step=5)
        wd.set_para(**cls.WINDOW_PARAMETER)
        return wd
