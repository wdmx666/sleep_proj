from recognition_pre_fog.feature_extraction import SignalFeatureExtractor,ExtractorManager
from recognition_pre_fog.extract_transform_load import SignalMarkNamedDataFrameLoader,SignalMarkDataLoader
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,MaxAbsScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.compose import TransformedTargetRegressor
import pandas


em = ExtractorManager().make_extractors()
print("em.get_extractor('F01')", em.get_extractor('F01'))
sfey = SignalFeatureExtractor(em.get_extractor('Identity'), groupby='sample_name',keep_group=True)
sfe1 = SignalFeatureExtractor(em.get_extractor('F01'), groupby='sample_name')
sfe2 = SignalFeatureExtractor(em.get_extractor('F02'), groupby='sample_name')
sfe3 = SignalFeatureExtractor(em.get_extractor('F03'), groupby='sample_name')

#data = SignalMarkNamedDataFrameLoader().load(data_path=r'E:\my_proj\fog_recognition\data\SignalMark')

cols = """A100_accelerometer_x	A100_accelerometer_y	A100_accelerometer_z""".split()+['sample_name']

data = SignalMarkDataLoader().load(data_path=r'E:\my_proj\fog_recognition\data\SignalMark')
print(data.columns)
print(data.values.shape)

ct = ColumnTransformer([('sfe1', sfe1, cols), ('sfe2', sfe2, cols), ('sfey', sfey, ['status', 'sample_name'])])
#cty = ColumnTransformer([('sfey', sfey,['status'])])
# lr = LogisticRegression()
# pipe = make_pipeline(ct, lr)
#
# print(pipe.get_params())
pipe =make_pipeline(ct)
#res = ct.fit(data, None).transform(data)
res = pipe.fit(data).transform(data)
pandas.DataFrame(data=res).to_csv("debug22.csv")

# parameters = [{'columntransformer__sfe1__step': [20], 'columntransformer__sfe2__step': [20]},
#               {'columntransformer__sfe1__step': [40], 'columntransformer__sfe2__step': [40]},
#               {'columntransformer__sfe1__step': [60], 'columntransformer__sfe2__step': [60]}]
# gs = GridSearchCV(estimator=pipe, param_grid=parameters)  # GridSearchCV 不方便使用
#
# regr_trans = TransformedTargetRegressor(regressor=gs, transformer=cty)  # 只能用于回归
#
# regr_trans.fit(data, data['status'])
print(res.shape)