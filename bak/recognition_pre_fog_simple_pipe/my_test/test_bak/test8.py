"""
@Name:        test8
@Description: ''
@Author:      Lucas Yu
@Created:     2018-11-06
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
from typing import Optional


def show(*x):
    print(x.__class__)
    #assert 1==2
    print(x)

for i in range(5):
    a: Optional[str]='a'
    b = a+"12"

import pandas
#from .algo_core import parse_dir
from sklearn.pipeline import Pipeline
from recognition_pre_fog.extract_transform_load._transform import RawDataETLInit



RAW_DATA_PATH = "E:/my_proj/fog_recognition/ExtendFoGData/raw_data"
SAMPLE_GROUPS = ["20161223","20161226","20161227","20180320","20180331","sample1"]

ret = Pipeline([('av', RawDataETLInit()), ('cd',None)])
#ret.fit('12','yyy')
#print("===================================")
df:pandas.DataFrame = ret.transform((RAW_DATA_PATH, SAMPLE_GROUPS))
df.to_csv('filenames.csv')

#ret.transform(12)
#print(ret.get_params())