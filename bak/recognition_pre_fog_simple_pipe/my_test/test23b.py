import pandas
from sklearn.pipeline import make_pipeline,Pipeline, FeatureUnion  # 二者是相对的，与ColumnTransformer不一样
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy,itertools

from recognition_pre_fog.preprocessing import SignalStatusReMark, dynamic_scale_data,filter_data
from recognition_pre_fog.feature_extraction import SignalFeatureExtractor, ExtractorManager
from recognition_pre_fog.commons.utils import retrieve_name

em = ExtractorManager().make_extractors()

data_path = r"E:\my_proj\fog_recognition\data\SignalMark\pd000001==201612231_1.csv"
df = pandas.read_csv(data_path)
feature_names = ["F01", "F02", "F03", "F04", "F05"]
numeric_cols = """A100_accelerometer_x	A100_accelerometer_y	A100_accelerometer_z	A100_gyroscope_x	A100_gyroscope_y	A100_gyroscope_z"""
numeric_cols = numeric_cols.split()
no_signal_cols = """time10	label_start_time	label_end_time	gait_type	label_confidence	stimulate_type	start_time	end_time	unit_confidence	status	status_code	one_hot"""
no_signal_cols = no_signal_cols.split()


ssr = SignalStatusReMark()

fu = FeatureUnion([(it, SignalFeatureExtractor(em.get_extractor(it))) for it in feature_names])
fu_gen_cols = ["_".join(i) for i in itertools.product(numeric_cols, feature_names)]
filter_trans = FunctionTransformer(filter_data)
dynamic_scale_trans = FunctionTransformer(dynamic_scale_data)

num_pipe = Pipeline([('fu', fu), ('filter_trans', filter_trans), ('dynamic_scale_trans',dynamic_scale_trans)])

ct = ColumnTransformer([('pipe', num_pipe, numeric_cols), ("Identity", SignalFeatureExtractor(em.get_extractor("Identity")), no_signal_cols)])

BIG_PIPE = make_pipeline(ssr, ct)
res = BIG_PIPE.fit(df).transform(df)

print(pandas.DataFrame(data=res, columns=fu_gen_cols+no_signal_cols).to_csv('debug222.csv'))
for k,v in BIG_PIPE.get_params().items(): print(k,v,'\n')








