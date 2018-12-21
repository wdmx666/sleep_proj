"""
@Name:        test15
@Description: ''
@Author:      Lucas Yu
@Created:     2018-11-10
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
# import joblib
#
# from recognition_pre_fog.extract_transform_load.data_extractor import ExtractorMetaData
#
#
# emd = ExtractorMetaData()
# emd.transform()
# res = joblib.load(emd.get_output_destination())
# for k,v in res.items():
#     print(k,v)


class A:
    def __init__(self,a,b ):
        self.a = a
        self.b = b

    def ff(self):
        self._c = 56
        print(self.a,self.b,vars(self))


a = A(12,34)
a.ff()


