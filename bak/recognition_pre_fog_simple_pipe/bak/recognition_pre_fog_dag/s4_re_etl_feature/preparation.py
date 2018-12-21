# ---------------------------------------------------------
# Name:        msg protocol for data exchange
# Description: some fundamental component
# Author:      Lucas Yu
# Created:     2018-04-06
# Copyright:   (c) Zhenluo,Shenzhen,Guangdong 2018
# Licence:
# ---------------------------------------------------------

import os
from pyhocon import ConfigFactory
from ..commons.common import MyConfig
from ..client.proj_config import MyNode


class FeatureReETLConfig(MyConfig):
    path = '../recognition_pre_fog_dag/s4_re_etl_feature/config/para.conf'
    _d_conf = ConfigFactory.parse_file(os.path.abspath(path))
    print(_d_conf)
    STATUS_INFO_COLS = _d_conf["STATUS_INFO_COLS"]
    STATUS_VALUES = _d_conf["STATUS_VALUES"]
    WEIGHTS = _d_conf['WEIGHTS']
    COLS_NAME = _d_conf["COLS_NAME"]

    @classmethod
    def scaleFeatureOneByOneService(cls):
        from .services import ScaleFeatureOneByOneService
        sv = ScaleFeatureOneByOneService()
        sv.set_para_with_prop({"info_cols": cls.COLS_NAME['info_cols']})
        return sv

    @classmethod
    def scaleFeatureOneByOne(cls, name=MyNode.ScaleFeatureOneByOne.name, dependencies=MyNode.RemarkScaleFeature.name, reset=False):
        from .controllers import ScaleFeatureOneByOne
        processor = ScaleFeatureOneByOne(name=name, dependencies=dependencies, reset=reset)
        processor.set_para_with_prop({"ScaleFeatureOneByOneService": cls.scaleFeatureOneByOneService()})
        return processor



