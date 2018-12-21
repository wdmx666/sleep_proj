"""
@Name:        services
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/2
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import path
import collections
import os
from sklearn import metrics

os.environ['R_USER'] = r'D:\ProgramLanguageCore\Python\anaconda351\Lib\site-packages\rpy2'
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from .algo_core import Utils
from ..commons.common import MyCalculator
import numba


# 单次验证
class TrainTestSplitValidator(MyCalculator):
    def __init__(self, name=None):
        super(TrainTestSplitValidator, self).__init__(name)
        self.para.setdefault("data_maker", None)
        self.para.setdefault("model", None)
        self.para.setdefault("strategy", None)


    def calculate(self, msg):
        """针对每一个患者样本,采用留一法训练样本和测试样本，返回样本预测结果包括状态和其概率"""
        df_train = self.para["data_maker"].prepare_train_df(msg[1])
        df_val = self.para["data_maker"].prepare_val_df(msg[0])
        cols = self.para["data_maker"].out_cols
        clf = self.para["model"]

        print(self.name, "数据准备完毕，进行拟合及评价")
        clf.fit(df_train[cols['feature_cols']], df_train[cols['target_cols']].values.reshape(-1), sample_weight=df_train[cols['weight_cols']].values.reshape(-1))

        result_df = df_val[["time10"] + cols['target_cols']]
        result_df.loc[:, "predict_status"] = clf.predict(df_val[cols['feature_cols']])
        proba = clf.predict_proba(df_val[cols['feature_cols']])

        result_df.loc[:, "predict_status_proba1"] = [i[0] for i in proba]
        result_df.loc[:, "predict_status_proba2"] = [i[1] for i in proba]

        msg = {"data": result_df, "filename": path.Path(msg[0][0])}  # 将作为验证的文件继续文件名记下通过返回值往下传递
        result_df = self.para["strategy"].mark_result(msg["data"])
        print(metrics.classification_report(result_df["status01"], result_df["filtered_predict_status01"]))

        result = dict()
        result["raw_f1_score"] = metrics.f1_score(result_df[cols['target_cols'][0]], result_df["predict_status"], average='micro')
        result["filtered_f1_score"] = metrics.f1_score(result_df["status01"],result_df["filtered_predict_status01"], average='micro')
        result["raw_kappa"] = metrics.cohen_kappa_score(result_df[cols['target_cols'][0]], result_df["predict_status"])
        result["filtered_kappa"] = metrics.cohen_kappa_score(result_df["status01"], result_df["filtered_predict_status01"])
        result["train_score"] = clf.score(df_train[cols['feature_cols']], df_train[cols['target_cols']].values.reshape(-1),
                                          sample_weight=df_train[cols['weight_cols']].values.reshape(-1))
        result["validation_score"] = clf.score(df_val[cols['feature_cols']], df_val[cols['target_cols']].values.reshape(-1),
                                               sample_weight=df_val[cols['weight_cols']].values.reshape(-1))
        result["predict_result"] = result_df


class CVProtocol(object):

    def split(self, samples_dir):
        return self.__leave_one_patient_out(samples_dir)

    def __leave_one_patient_out(self, samples_dir):
        p = path.Path(samples_dir)
        print("sample_dir", samples_dir)
        f4model = []
        fp = np.array([p.joinpath(i) for i in p.files()])
        for i in range(len(fp)):
            fpr = np.roll(fp, -i)
            f4model.append(([fpr[0]], list(fpr[1:])))
        return f4model


class CrossValidator(MyCalculator):
    def __init__(self, name=None):
        super().__init__(name)
        self.para.setdefault("TrainTestSplitValidator", None)
        self.para.setdefault("CVProtocol", None)

    @numba.jit
    def calculate(self, samples_dir):
        result = dict()  # one request result
        items = collections.deque(self.para["CVProtocol"].split(samples_dir))
        #print(self.name, items)
        # 注意items的每一项item都是列表的元组，即item的每项都是列表

        while items:
            item = items.popleft()
            print(self.name, item)
            result[path.Path(item[0][0]).basename().__str__()] = self.para['TrainTestSplitValidator'].calculate(item)
            print('<<<<==========>>>>')
        #print("one_cv_result: ", result)
        return result