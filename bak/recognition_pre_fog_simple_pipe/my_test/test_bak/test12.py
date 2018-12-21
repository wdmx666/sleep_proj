"""
@Name:        test12
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/9
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
from pyhocon import ConfigFactory
from sklearn.pipeline import make_pipeline
import os
from recognition_pre_fog.extract_transform_load._transform import RawDataETLInit,SignalDataExtractor,DropDuplicateFillLostTimePoint
from recognition_pre_fog.extract_transform_load._extractor import create_part_full_field_map
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from recognition_pre_fog.extract_transform_load._extractor import filter_data


data_description_path = r"E:\my_proj\fog_recognition\recognition_pre_fog_pipe\recognition_pre_fog\conf\data_description.conf"
_d_conf = ConfigFactory.parse_file(os.path.abspath(data_description_path))

rd = RawDataETLInit()
para = dict(part_fields=create_part_full_field_map(_d_conf['PART_FIELD']), drop_cols=_d_conf['DROP_COLS'])
pipe = make_pipeline(SignalDataExtractor(**para), DropDuplicateFillLostTimePoint())


#imputer = SimpleImputer(missing_values=None,strategy='median')
#imputer = SimpleImputer(missing_values=None,strategy='constant',fill_value=10)
filter_1 = ColumnTransformer([('filter_data', FunctionTransformer(filter_data, validate=True), _d_conf['SIGNAL_COLS_NAME']['data'])], remainder='drop')
imputer_1 = ColumnTransformer([('imputer', SimpleImputer(), slice(0, -1))], remainder='passthrough')
#pipe2= Pipeline([('a',wt), ('b',ddft)])
pipe2 = make_pipeline(filter_1, imputer_1)
res = rd.transform(_d_conf)
for it in res:
    df=pipe.transform(it)
    #print(pipe2.transform(it))
    #print(ProjectConfig.preprocessor().fit_transform(df)[0:4])
    print(pipe2.fit_transform(df))
