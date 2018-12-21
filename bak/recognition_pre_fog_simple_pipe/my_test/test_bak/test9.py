
import sklearn
import pandas as pd
import numpy
import pprint
p = r"E:\my_proj\fog_recognition\pre_fog_v3\resources\fixed_data\TidyData\formed_signal\20161223_1_1.csv"
df = pd.read_csv(p)

from sklearn import preprocessing
df1=df.iloc[0:10,0:2]
#quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
#res = quantile_transformer.fit_transform(df1)
from pyhocon import ConfigFactory

_d_conf = ConfigFactory.parse_file(r'E:\my_proj\fog_recognition\recognition_pre_fog_pipe\recognition_pre_fog\conf\data_description.conf')

from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline



def filter_data(x: numpy.ndarray)->numpy.ndarray:
    from typing import List
    import pandas

    def between_range(array: numpy.ndarray)-> List[float]:
        interval = pandas.Interval(round(numpy.quantile(array, 0.005),2), round(numpy.quantile(array, 0.995),2), closed='both')
        print('-->', interval)
        return list(map(lambda y: y if y in interval else None, array))
    result = [between_range(x[:, it]) for it in range(x.shape[1])]
    result = numpy.array(result).T
    return result


filter_f = FunctionTransformer(filter_data, validate=True)


#imputer = SimpleImputer(missing_values=None,strategy='median')
#imputer = SimpleImputer(missing_values=None,strategy='constant',fill_value=10)
filter_1 = ColumnTransformer([('filter_data', filter_f, _d_conf['SIGNAL_COLS_NAME']['data'])], remainder='drop')
#imputer_1 = ColumnTransformer([('imputer', imputer, slice(0, -1))], remainder='passthrough')
#pipe = Pipeline([('filter_1',filter_1),('imputer_1', imputer_1)])

#numeric_features = _d_conf['SIGNAL_COLS_NAME']['data']
numeric_transformer = Pipeline([('filter_data', filter_f), ('imputer', SimpleImputer(strategy='median'))])
#preprocessor = ColumnTransformer([('num', numeric_transformer, numeric_features)], remainder='drop')

#print(_d_conf['SIGNAL_COLS_NAME']['data'].__len__(),df.columns)

#ct.fit(df)
res = filter_1.fit_transform(df)
print(res.head())

#res = preprocessor.fit_transform(df)
#print(filter_f.__dir__())
#print(preprocessor.named_transformers_['num'].get_feature_names())
#print(preprocessor.get_feature_names())#.named_steps['filter_data'].get_feature_names())
#print(df.shape,type(res),res.shape)
#print(pd.DataFrame(res))
#print(pd.DataFrame(res).dropna().shape)
