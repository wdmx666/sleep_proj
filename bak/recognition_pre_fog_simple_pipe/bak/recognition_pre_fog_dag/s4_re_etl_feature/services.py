"""
@Name:        services
@Description: ''
@Author:      Lucas Yu
@Created:     2018/10/31
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import pandas as pd
import path
from concurrent import futures
from sklearn import preprocessing
import numba

from ..client.proj_config import MyNode
from ..commons.common import MyCalculator


# 根据条件对数据进行填充
class FeatureDataETL:
    """根据多个视频数据填充"""
    @staticmethod
    def clean_feature(df_all, info_cols):
        con = pd.Series([True]*df_all.shape[0])
        for col in df_all.columns:
            if col not in info_cols:
                con = con&(df_all[col] < 10**9)&(df_all[col]>-10**9)
        df_all = df_all[con]
        return df_all.fillna(df_all.mean())

    @staticmethod
    @numba.jit()
    def merge_2_big_df(dir_path,new_col):
        filenames = path.Path(dir_path).files()
        df_all = pd.DataFrame()
        for fn in filenames:
            df = pd.read_csv(open(fn))
            df[new_col] = fn.basename()
            df_all = df_all.append(df, ignore_index=True)
        return df_all

    @staticmethod
    def make_col_scale_map(df_all,info_cols):
        df_all_drop = df_all.drop(columns=info_cols)
        col_treat = {}

        for i in df_all_drop.columns:
            if abs(df_all_drop[i].max()/df_all_drop[i].min())>5000:
                col_treat[i] = preprocessing.quantile_transform
            else:
                col_treat[i] = preprocessing.scale
        return col_treat

    @staticmethod
    def scale_df(df_one_file, col_treat, info_cols):
        df_drop = df_one_file.drop(columns=info_cols)
        for j in df_drop.columns:
            df_one_file[j] = col_treat[j](df_one_file[j].values.reshape(-1, 1), copy=True)
        return df_one_file


# focus on business logic not the detail
class ScaleFeatureOneByOneService(MyCalculator):
    def __init__(self, name=None):
        super(ScaleFeatureOneByOneService, self).__init__(name)
        self.para.setdefault("info_cols", None)

    def calculate(self, msg):
        new_col = "sample_name"
        df_all = FeatureDataETL.merge_2_big_df(msg, new_col)
        info_cols = [i for i in self.para.get("info_cols") if i in df_all.columns] + [new_col]
        df_all = FeatureDataETL.clean_feature(df_all, info_cols)
        print("clean_feature over")
        col_treat = FeatureDataETL.make_col_scale_map(df_all, info_cols)

        res_df_map = {}
        with futures.ProcessPoolExecutor(max_workers=4) as executor:
            future_to_it = {}
            for it in df_all[new_col].unique():
                df_one_file = df_all[df_all[new_col] == it]
                future_to_it[executor.submit(FeatureDataETL.scale_df, df_one_file, col_treat, info_cols)] = it
            for future in futures.as_completed(future_to_it):
                it = future_to_it[future]
                res_df_map[it] = future.result()
                print("scale_df over")
        # 单线程成运行
        # for it in df_all[new_col].unique():
        #     df_one_file = df_all[df_all[new_col] == it]
        #     res_df_map[it] = FeatureDataETL.scale_df(df_one_file, col_treat, info_cols)
        #     print("scale_df over")

        return res_df_map

