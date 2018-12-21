# ---------------------------------------------------------
# Name:        command for client start
# Description: some fundamental entry-point
# Author:      Lucas Yu
# Created:     2018-05-03
# Copyright:   (c) Zhenluo,Shenzhen,Guangdong 2018
# Licence:
# ---------------------------------------------------------

import path

from ..client.proj_config import MyNode
from ..commons.common import MyProcessor
from ..commons.helper import ProcessorHelper


#  若不将service与controller分离将导致service不能重用
class ScaleFeatureOneByOne(MyProcessor):
    def __init__(self, name=None, dependencies=None,reset=False):
        super(ScaleFeatureOneByOne, self).__init__(name, dependencies,reset)
        self.para.setdefault("ScaleFeatureOneByOneService", None)
        if not self.dependencies:
            self.dependencies.append(MyNode.RemarkScaleFeature.name)

    def prepare(self):
        files_name = ProcessorHelper.pair_input_by_id({node: path.Path(data_url) for node, data_url in self.request.items()})
        processing_request = {idx: it for idx, it in files_name.items() if not self.output(name_info=it[self.dependencies[0]].basename()).exists()}
        if not processing_request:
            self.finished = True
            print(self.name, '的结果已经缓存，无需再次计算！')
        self.request = {node: path.Path(data_url) for node, data_url in self.request.items()}

    def output(self, name_info=None, new_name=None):
        return path.Path(self.get_output_destination().joinpath(name_info))

    def process(self):
        print("=================================================")
        res_df = self.para["ScaleFeatureOneByOneService"].calculate(self.request[self.dependencies[0]])
        for it, df in res_df.items():
            df.to_csv(self.output(name_info=it), index=False)
