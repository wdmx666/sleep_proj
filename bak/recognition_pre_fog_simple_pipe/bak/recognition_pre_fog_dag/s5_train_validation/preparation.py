"""
@Name:        preparation
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/4
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
from ..client.proj_config import MyNode
from pyhocon import ConfigFactory
import os

from recognition_pre_fog_dag.commons.common import MyConfig, MyProperties

class ValidationConfig(MyConfig):
    conf_path = '../recognition_pre_fog_dag/s5_train_validation/config/para.conf'
    _d_conf = ConfigFactory.parse_file(os.path.abspath(conf_path))
    STATUS_VALUES = _d_conf['STATUS_VALUES']

    COLS_LIST = {'feature_cols': _d_conf['INPUT_COLS']['feature_cols'].split(),
                 'target_cols': _d_conf['INPUT_COLS']['target_cols'].split(),
                 'weight_cols': _d_conf['INPUT_COLS']['weight_cols'].split(),
                 'info_cols': _d_conf['INPUT_COLS']['info_cols'].split()}


    @classmethod
    def dataMaker4Model(cls):
        from .algo_core import DataMaker4Model
        dm = DataMaker4Model()
        dm.set_para_with_prop({"input_cols": cls.COLS_LIST, "status_values": cls.STATUS_VALUES})
        cls.check_object(dm)
        return dm

    @classmethod
    def mlModel(cls):
        from sklearn.ensemble import RandomForestClassifier
        para = dict(n_estimators=100, max_depth=2, max_features=0.2, class_weight="balanced",
                    min_samples_split=6, min_samples_leaf=4, random_state=0, n_jobs=6)
        clf = RandomForestClassifier()
        clf.set_params(**para)
        return clf

    @classmethod
    def trainTestSplitValidator(cls):
        from .services import TrainTestSplitValidator
        from .algo_core import StrategyResult2

        ttsv = TrainTestSplitValidator()
        ttsv.set_para_with_prop({"data_maker": cls.dataMaker4Model(), "model": cls.mlModel(),"strategy": StrategyResult2()})
        cls.check_object(ttsv)
        return ttsv

    @classmethod
    def crossValidator(cls):
        from .services import CVProtocol, CrossValidator
        protocol = CVProtocol()
        vd = CrossValidator()
        vd.set_para_with_prop({"TrainTestSplitValidator": cls.trainTestSplitValidator(), "CVProtocol":protocol})
        return vd

    @classmethod
    def validationProcessor(cls, name=MyNode.ValidationProcessor.name, dependencies=MyNode.ScaleFeatureOneByOne.name,reset=False):
        from .controllers import ValidationProcessor
        vp = ValidationProcessor(name, dependencies, reset)
        vp.set_para_with_prop({'CrossValidator':cls.crossValidator()})
        return vp