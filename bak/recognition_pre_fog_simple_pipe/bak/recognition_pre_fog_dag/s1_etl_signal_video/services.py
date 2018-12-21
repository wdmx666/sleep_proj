# ---------------------------------------------------------
# Name:        algorithm for etl signal
# Description: algorithm for etl signal
# Author:      Lucas Yu
# Created:     2018-04-03
# Copyright:   (c) Zhenluo,Shenzhen,Guangdong 2018
# Licence:
# ---------------------------------------------------------

from ..commons.common import MyCalculator
from .algo_core import WorkbookMerger, CoordinateTransformer, DataFilter, DataFiller, VideoMarkParser


class SignalETLService(MyCalculator):

    def __init__(self, name=None):
        super().__init__(name)

        self.para.setdefault("parts", None)
        self.para.setdefault("part_full_field_map", None)
        self.para.setdefault("tm_path", None)


    def set_para_with_prop(self, my_props):
        self.para.update(my_props)

    def calculate(self, msg):
        print('msg-->', msg)
        parts = [part for part in self.para.get("parts") if part not in ["C101", "C201"]]
        wb = WorkbookMerger(self.para.get("parts"), self.para.get("part_full_field_map"))
        ct = CoordinateTransformer(self.para["tm_path"][msg['id']], parts)
        #wa = WeightCalculator(self.para['col_list'])
        data_filter = DataFilter()  # 类似service
        data_filler = DataFiller()
        df = wb.arrange_workbook(msg['signal_file'])
        df = ct.transform(df)
        df = data_filter.transform(df)
        df = data_filler.transform(df)
        #df = wa.transform(df)
        return df



