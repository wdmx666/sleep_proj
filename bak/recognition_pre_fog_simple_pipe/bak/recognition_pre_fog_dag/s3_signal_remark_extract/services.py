"""
@Name:        services
@Description: 特征提取，由多个基本的可以插件替换的服务组成
@Author:      Lucas Yu
@Created:     2018/10/24
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import copy
import pandas as pd
from sklearn import preprocessing
from ..commons.common import MyCalculator, MyProperties
from .algo_general import OneSignalFeatures


# 由于各个模块之间强关联性，因此不能像web的service那样独立。
class StatusReMarkService(MyCalculator):
    def __init__(self, name=None):
        super(StatusReMarkService, self).__init__(name)
        self.para = MyProperties()
        self.para.setdefault("status_definition", None)
        self.para.setdefault("pre_fog_time_len", None)

    def set_para_with_prop(self, my_props):
        self.para.update(my_props)

    def __mark_pre_fog_according_to_fog(self, df):
        df.loc[:, 're_status'] = df['status']
        # 使用二元窗找到切换点
        def get_change_point(dfs):
            status_change = []
            for idx in dfs.head(dfs.shape[0] - 1).index:
                if dfs.loc[idx, 'status'] != 'fog' and dfs.loc[idx + 1, 'status'] == 'fog':
                    status_change.append(idx)
            return status_change

        # 根据切换点和定义pre_fog时间长标记
        def re_status(dfs, change_point, time_len):
            con = dfs.loc[change_point, 'time10'] - time_len
            for i in range(change_point, -1, -1):
                if dfs.loc[i, 'time10'] < con or dfs.loc[i, 'status'] == 'fog':
                    break
                else:
                    dfs.loc[i, 're_status'] = 'pre_fog'
            return dfs

        status_changes = get_change_point(df)
        for it in status_changes:
            df = re_status(df, it, self.para["pre_fog_time_len"])
        return df

    @staticmethod
    def __mark_status(it, STATUS_DEFINITION, status="normal"):
        for k, lst in STATUS_DEFINITION.items():
            if it in lst:
                status = k
                break
        return status

    def calculate(self, msg):  # mark the "pre_fog" status according to the "fog"
        df = pd.read_csv(msg)
        df.loc[:, 'status'] = df['gait_type'].apply(self.__mark_status, args=(self.para["status_definition"],))
        return self.__mark_pre_fog_according_to_fog(df)


# 将数据标准化
class ScaleService(MyCalculator):
    def __init__(self, name=None):
        super(ScaleService, self).__init__(name)
        self.para.setdefault("signal_cols", None)

    def calculate(self, df):
        scaler = preprocessing.StandardScaler()
        df_signal = df[self.para['signal_cols']]
        df_res_ = pd.DataFrame(scaler.fit_transform(df_signal), columns=self.para['signal_cols'], index=df_signal.index)
        return df_res_.join(df.drop(columns=self.para['signal_cols']), how="inner").reset_index(drop=True), scaler


class FeatureService(MyCalculator):
    def __init__(self, name=None):
        super(FeatureService, self).__init__(name)
        self.para.setdefault("signal_cols", None)
        self.para.setdefault("features", None)
        self.para.setdefault("window", None)

    def calculate(self, df):
        all_fields_feature = df.drop(columns=self.para['signal_cols'])
        for field in self.para["signal_cols"]:
            ob = OneSignalFeatures(self.para["features"], df[field]).set_window(self.para['window'])
            field_feature_df = ob.cal_features(copy.deepcopy(df[["time10"]]))
            all_fields_feature = all_fields_feature.merge(field_feature_df, how="inner", on="time10")
        return all_fields_feature

        # if "status" in all_fields_feature.columns:
        #     all_fields_feature = all_fields_feature[
        #         all_fields_feature["status"].isin(self.para['status_values'])]  # 选择出目标状态的数据
        # all_fields_feature.to_csv(self.get_output_destination(), index=False)


class RemarkScaleFeatureService(MyCalculator):
    def __init__(self, name=None):
        super(RemarkScaleFeatureService, self).__init__(name)
        self.para.setdefault("StatusReMarkService", MyCalculator())
        self.para.setdefault("ScaleService", MyCalculator())
        self.para.setdefault("FeatureService", MyCalculator())

    def calculate(self, msg):
        re_mark_sv = self.para["StatusReMarkService"]
        scale_sv = self.para["ScaleService"]   # 每个信号分别进行标准化
        feature_sv = self.para["FeatureService"]
        df = re_mark_sv.calculate(msg)
        df = scale_sv.calculate(df)[0]
        df = feature_sv.calculate(df)
        return df