"""
@Name:        controllers
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/2
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
from ..commons.common import MyProcessor
import path
import joblib


class ValidationProcessor(MyProcessor):  # only封装逻辑
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)
        self.para.setdefault('CrossValidator', None)
        self.para.setdefault('param_id', None)

    def output(self, name_info=None, new_name=None):
        return path.Path(self.get_output_destination().joinpath(name_info))

    def prepare(self):
        self.request = {node: path.Path(data_url) for node, data_url in self.request.items()}
        print("======================>>>>>>>>>>",self.para, self.request)

    def process(self):  # 到这一步已经只需要指定的输入数据和参数进行计算了，其他逻辑它们不管
        print(self.class_name, self.request[self.dependencies[0]])
        if not self.output(self.para['param_id']).exists():
            result = self.para["CrossValidator"].calculate(self.request[self.dependencies[0]])
            joblib.dump(result, self.output(self.para['param_id']))