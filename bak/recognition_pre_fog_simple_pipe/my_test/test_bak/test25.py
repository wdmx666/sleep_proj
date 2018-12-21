import pandas
from sklearn.pipeline import make_pipeline, Pipeline, FeatureUnion  # 二者是相对的，与ColumnTransformer不一样
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import itertools,functools
from typing import List
from joblib import Parallel,delayed



data_path = r"E:\my_proj\fog_recognition\recognition_pre_fog_simple_pipe\data\SignalMarker\pd000001==20161223_1_1.csv"

df = pandas.read_csv(data_path)


def select_data(df:pandas.DataFrame)->pandas.DataFrame:
    print(df.__class__)
    df=df[df['gait_type']=='normal']
    print(df.columns)
    return df

ft = FunctionTransformer(select_data)

res = ft.transform(df.iloc[:,2:8])
print(res.__class__)