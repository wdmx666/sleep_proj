
from recognition_pre_fog.batch_transform import BatchTransformer
from recognition_pre_fog.wrapper_estimator import WrapperEstimator
from recognition_pre_fog.preprocessing._signal_mark import DataSelector
from recognition_pre_fog.extract_transform_load import SignalMarkDataLoader
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
import time

p = r"E:\my_proj\fog_recognition\recognition_pre_fog_simple_pipe\data\SignalMarker"

data = SignalMarkDataLoader().load(p)
#data = data.sample(8000)
#data =data.reset_index(drop=True)


btf = BatchTransformer()
bts = DataSelector()
we = WrapperEstimator(RandomForestClassifier())

cachedir = r'E:\my_proj\fog_recognition\recognition_pre_fog_simple_pipe\data\cache'
pipe = make_pipeline(btf, bts, we)
pipe.fit(data)
#print(pipe)
time.sleep(4)
print("================================================================================================================")
print("----------------->>>>>>>>>>>>>>>>主线程开始使用管子去预测！！！！<<<<<<<<<<<<<<<<-------------------")
res = pipe.predict(data)
print(res)



