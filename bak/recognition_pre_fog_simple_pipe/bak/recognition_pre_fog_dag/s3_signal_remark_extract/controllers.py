# ---------------------------------------------------------
# Name:        msg protocol for data exchange
# Description: some fundamental component
# Author:      Lucas Yu
# Created:     2018-04-07
# Copyright:   (c) Zhenluo,Shenzhen,Guangdong 2018
# Licence:
# ---------------------------------------------------------

import path
import joblib
from concurrent import futures
from ..client.proj_config import MyNode
from ..commons.helper import ProcessorHelper
from ..commons.common import MyProperties

from ..commons.common import MyProcessor


# 数据整理各种步骤排序的策略
data_processor_map = {"1":"Scale", "2":"Feature", "3":"Select"}
router = ["123", "132", "213", "231", "312", "321"]


# 调用一个任务时该任务如何执行，先检查缓存，缓存有则直接返回地址，否则再执行返回
# luigi中调用任务总是返回其输出地址
class RemarkScaleFeature(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        super(RemarkScaleFeature, self).__init__(name, dependencies, reset)
        self.para = MyProperties()
        if not self.dependencies:
            self.dependencies.append(MyNode.SignalMark.name)
        self.para.setdefault("RemarkScaleFeatureService", None)

    def output(self, name_info=None, new_name=None):
        return self.get_output_destination().joinpath(name_info)

    def prepare(self):  # 尽量避免逻辑交叉
        print(self.dependencies)
        """1.获取前驱(依赖)任务输出文件的地址作为输入；2.并检查本任务的输出是否已经存在，若存在则不调用处理方法"""
        files_name = ProcessorHelper.pair_input_by_id({node: path.Path(
            data_url) for node, data_url in self.request.items()})
        self.request = {idx: it for idx, it in files_name.items() if not self.output(name_info=it[self.dependencies[0]].basename()).exists()}

    def process(self):
        # with futures.ProcessPoolExecutor(max_workers=4) as executor:
        #     future_to_it = {executor.submit(self.para["RemarkScaleFeatureService"].calculate, it[self.dependencies[0]]): it
        #                     for idx, it in self.request.items()}
        #     for future in futures.as_completed(future_to_it):
        #         it = future_to_it[future]
        #         future.result().to_csv(self.output(name_info=it[self.dependencies[0]].basename()), index=False)

        #for idx, it in self.request.items():
        fn=self.para["RemarkScaleFeatureService"].calculate
        with joblib.Parallel(n_jobs=4) as parallel:
            #for idx, it in self.request.items():
            future_result = parallel(joblib.delayed(fn)(it[self.dependencies[0]]) for idx, it in self.request.items())

            #future_result = self.para["RemarkScaleFeatureService"].calculate(it[self.dependencies[0]])
            c=0
            for idx, it in self.request.items():
                future_result[c].to_csv(self.output(name_info=it[self.dependencies[0]].basename()), index=False)
                c+=1









