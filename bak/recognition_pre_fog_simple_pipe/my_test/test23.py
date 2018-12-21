import pandas
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

from recognition_pre_fog.preprocessing import SignalStatusReMark,dynamic_scale_data,filter_data
from recognition_pre_fog.feature_extraction import SignalFeatureExtractor,ExtractorManager

em = ExtractorManager().make_extractors()

data_path = r"E:\my_proj\fog_recognition\data\SignalMark\pd000001==201612231_1.csv"
df = pandas.read_csv(data_path)
feature_names = ["F01", "F02", "F03", "F04", "F05"]
numeric_cols = """A100_accelerometer_x	A100_accelerometer_y	A100_accelerometer_z	A100_gyroscope_x	A100_gyroscope_y	A100_gyroscope_z"""
numeric_cols = numeric_cols.split()
no_signal_cols = """time10	label_start_time	label_end_time	gait_type	label_confidence	stimulate_type	start_time	end_time	unit_confidence	status	status_code	one_hot"""
no_signal_cols = no_signal_cols.split()


ssr = SignalStatusReMark()
col_trans = [(it, SignalFeatureExtractor(em.get_extractor(it)), numeric_cols) for it in feature_names]
#identity_trans = [("Identity", SignalFeatureExtractor(em.get_extractor("Identity")), no_signal_cols)]

ct = ColumnTransformer(col_trans)

filter_data_trans = FunctionTransformer(filter_data)
dynamic_scale_data_trans = FunctionTransformer(dynamic_scale_data)

comp = [(str(it).lower(), it) for it in [ssr, ct, filter_data_trans, dynamic_scale_data_trans]]
pipe = Pipeline(comp)
res = pipe.fit(df).transform(df)
ct_big = ColumnTransformer(pipe,)

print(res[1])






