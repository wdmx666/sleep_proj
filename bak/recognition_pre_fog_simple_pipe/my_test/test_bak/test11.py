from pyhocon import ConfigFactory
from typing import Dict,List
import joblib
import os


def create_part_full_field_map(part_field):
    time_field = ["time16", "time10", "hour", "minute", "second", "ms"]
    return {part: time_field + [part + "_" + field for field in part_field[part] if field not in time_field] for part in
            part_field}


def make_input_item(raw_data:List[Dict[str,str]],transform_matrix:Dict[str,str], sheet_part:Dict[str,List[str]])->List[Dict[str,str]]:
    """{'id': 'pd000001','part': [a,b,c], 'transform_matrix':'path', 'signal_file': 'path','video_result_file': 'path',
        'data_group': '20161223', 'sample_times': '1_1'} 记录数据的数据结构说明"""
    full_description = []
    for it in raw_data:
        description = {}
        description['part'] = sheet_part[it['id']]
        description['transform_matrix'] =transform_matrix[it['id']]
        description.update(it)
        full_description.append(description)
    return full_description


path = r"E:\my_proj\fog_recognition\recognition_pre_fog_pipe\recognition_pre_fog\conf\data_description.conf"
_d_conf = ConfigFactory.parse_file(os.path.abspath(path))
SHEET_PART = _d_conf["SHEET_PART"]
PARTS = _d_conf["PARTS"]
PART_FIELD = create_part_full_field_map(_d_conf["PART_FIELD"])
TRANSFORM_MATRIX= _d_conf["TRANSFORM_MATRIX"]



print(PART_FIELD)

print({k:SHEET_PART.get(k) for k in SHEET_PART})
print({k:TRANSFORM_MATRIX.get(k) for k in TRANSFORM_MATRIX})
df0 = joblib.load(r'E:\my_proj\fog_recognition\recognition_pre_fog_pipe\recognition_pre_fog\conf\raw_data')

res=make_input_item(df0.to_dict('records'),TRANSFORM_MATRIX,SHEET_PART)
for it in res:
    print(it)