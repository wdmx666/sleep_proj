"""
@Name:        test24
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/15
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
from recognition_pre_fog.batch_transform import BatchTransformer
from recognition_pre_fog.extract_transform_load import SignalMarkNamedDataFrameLoader,SignalMarkDataLoader

if __name__ == "__main__":
    data = SignalMarkDataLoader().load(data_path=r'E:\my_proj\fog_recognition\recognition_pre_fog_simple_pipe\data\SignalMarker')
    btf = BatchTransformer()
    res = btf.transform(data)
    print(res.to_csv('debug.csv'))