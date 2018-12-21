from recognition_pre_fog.batch_transform import BatchTransformer
from recognition_pre_fog.preprocessing._signal_mark import DataSelector
from recognition_pre_fog.extract_transform_load import SignalMarkDataLoader
from sklearn.pipeline import make_pipeline

p = r"E:\my_proj\fog_recognition\recognition_pre_fog_simple_pipe\data\SignalMarker"

data = SignalMarkDataLoader().load(p)

btf = BatchTransformer()
bts = DataSelector()

pipe = make_pipeline(btf, bts)
res = pipe.fit_transform(data)
print(res.head())



