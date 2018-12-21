# coding:UTF8#
import os
import path
from pyhocon import ConfigFactory
from ..client.proj_config import MyNode
from ..commons.common import MyConfig,MyProperties


# 配置静态数据(变化程度极低的输入)
# dependency inject
class SignalETLConfig(MyConfig):
    from ..commons.utils import PreProcedure
    path = "../recognition_pre_fog_dag/s1_etl_signal_video/config/para.conf"
    _d_conf = ConfigFactory.parse_file(os.path.abspath(path))
    part_field = _d_conf["SHEET_FIELD"]
    PARTS = _d_conf["PARTS"]
    SHEET_FIELD_MAP = PreProcedure.create_part_full_field_map(_d_conf["SHEET_FIELD"])
    TRANSFORM_MATRIX_PATH = _d_conf["TRANSFORM_MATRIX_PATH"]
    TRANSFORM_MATRIX_FILE = _d_conf["TRANSFORM_MATRIX_FILE"]
    SIGNAL_SAVE_PATH = _d_conf["SIGNAL_SAVE_PATH"]
    RAW_DATA_PATH = r'E:/my_proj/fog_recognition/ExtendFoGData/raw_data'


# #####一些公共的服务对象##################################################################################
    @classmethod
    def appGraph(cls):
        from ..commons.scheduler import AppGraph
        graph = AppGraph()
        return graph

    @classmethod
    def appSession(cls):
        from ..commons.scheduler import AppSession
        sess = AppSession()
        return sess

    @classmethod
    def signalETLInit(cls):
        from .controllers import SignalETLInit
        node = SignalETLInit(name=MyNode.SignalETLInit.name, reset=False)
        node.set_para_with_prop({'raw_data_path':cls.RAW_DATA_PATH})
        return node

    @classmethod
    def signalETLService(cls):
        from .services import SignalETLService
        sv = SignalETLService()
        etl_props = MyProperties()
        etl_props.update({"parts": cls.PARTS})
        etl_props.update({"part_full_field_map": cls.SHEET_FIELD_MAP})
        etl_props.update({'tm_path': SignalETLConfig.TRANSFORM_MATRIX_FILE})
        sv.set_para_with_prop(etl_props)
        return sv

    @classmethod
    def signalETL(cls):
        from .controllers import SignalETL
        processor = SignalETL(name=MyNode.SignalETL.name, dependencies=[MyNode.SignalETLInit.name], reset=False)
        processor.set_para_with_prop({"SignalETLService": cls.signalETLService(),'max_workers':4})
        return processor

    @classmethod
    def videoMarkETL(cls):
        from .controllers import VideoMarkETL
        processor = VideoMarkETL(name=MyNode.VideoMarkETL.name, dependencies=[MyNode.SignalETLInit.name])
        return processor

    @classmethod
    def signalMark(cls):
        from .controllers import SignalMark
        from .algo_core import SignalMarker
        processor = SignalMark(name=MyNode.SignalMark.name,dependencies=[MyNode.SignalETL.name,MyNode.VideoMarkETL.name])
        processor.set_para_with_prop({'SignalMarker': SignalMarker(),'max_workers':4})
        return processor







