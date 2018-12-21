"""
@Name:        controllers
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/4
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
from ..commons.common import MyProcessor


class MyEventParaValidator(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)
        self.para.setdefault('CV', None)
        self.para.setdefault('param_id', None)

    def process(self):
        pass
