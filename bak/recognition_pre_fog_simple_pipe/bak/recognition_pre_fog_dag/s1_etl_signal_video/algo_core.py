"""
@Name:        algo_core
@Description: ''
@Author:      Lucas Yu
@Created:     2018/10/22
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import numpy as np
import copy
import path
import os, glob
import pandas as pd
import keras
import itertools



# Excel 工作簿的融合
# 以下是提供信号处理服务的基本功能
# 与数据模型相关的部分
class WorkbookMerger:
    """
    # the function of the class is just to satisfy te demands.
    """
    # the attributes are used in the class not for outside
    # so they are set to be private

    def __init__(self, sheet_names, sheet_fields):
        """
        :param sheet_names: 列出每个数据簿的表格的名字
        :param sheet_fields: 列出每个表格的字段名字
        """
        self.__sheet_names = sheet_names
        self.__sheet_fields = sheet_fields

    def __read_and_rename(self, workbook_name):
        """
        :param workbook_name: 数据簿的名称（全名称：包括地址）
        :return: 返回一个含有多个df的map
        """
        print('工作簿名称：', workbook_name)
        df_map = pd.read_excel(workbook_name, sheet_name=self.__sheet_names, header=None)
        for part in self.__sheet_names:
            df_map[part].columns = self.__sheet_fields[part]
        return df_map

    def arrange_workbook(self, workbook_path):
        df_map=self.__read_and_rename(workbook_path)
        for key in df_map:
            df_map[key].drop(columns=set(df_map[key].columns) & set(["time16", "hour", "minute", "second", "ms"]),
                             inplace=True)
        result_df = pd.DataFrame()
        for i in range(len(self.__sheet_names)):
            df = df_map[self.__sheet_names[i]].drop_duplicates(subset="time10").reset_index(drop=True)

            if i != 0:
                result_df = result_df.merge(df, on="time10",how="inner")
            else:
                result_df = df
        return result_df

    def collect_method(self, **kwargs):
        print(kwargs["workbook_path"])


# 原始信号数据过滤，过滤策略具有很强的变动性
class DataFilter:
    def transform(self,df):
        p1 = df.quantile(q=0.005)
        p2 = df.quantile(q=0.995)

        res = []
        for i in df.columns:
            if i not in ["time10"]:
                res.append(df[i][(df[i] > p1[i]) & (df[i] < p2[i])])
        res.append(df["time10"])
        df = pd.concat(res, axis=1, join="inner")
        return df


# 填充数据，有的数据会有出现时间序列的不连续。
class DataFiller:
    def transform(self,df):
        df = df.reset_index(drop=True,)
        lost_row = []
        dtf = df["time10"].diff().dropna()
        dtf = dtf[dtf > 1]
        for idx, v in dtf.iteritems():
            avg = pd.Series.add(df.loc[idx], df.loc[idx - 1]) / 2
            # print(df["time10"].loc[idx-1],df["time10"].loc[idx])
            time_range = range(df["time10"].loc[idx - 1], df["time10"].loc[idx], 1)
            for i in range(1, len(time_range)):
                ds = copy.deepcopy(avg)
                ds["time10"] = int(time_range[i])
                lost_row.append(ds.astype(int))
                # print(time_range[i])
        return df.append(pd.DataFrame(lost_row), ignore_index=True).sort_values("time10").reset_index(drop=True)


# 坐标转换矩阵的加载
class CoordinateTransformer:
    def __init__(self, tm_path,sheet_names):
        self.__tm_path = tm_path
        self.__sheet_names = sheet_names

    def transform(self, df):
        df_tm = pd.read_csv(open(self.__tm_path))
        result_df = copy.deepcopy(df[["time10"]])
        all_cols = []
        for part in self.__sheet_names:
            tm_df = df_tm[df_tm["part"] == part]
            tm_df.drop(columns=["part"],inplace=True)
            cols = [part+"_"+col for col in tm_df.columns]
            all_cols.extend(cols)
            result_df = result_df.join(pd.DataFrame(np.matmul(df[cols].values,tm_df.values), columns=cols), how="inner")
        result_df = result_df.merge(df.drop(columns=all_cols),on="time10",how="inner")

        for col in result_df.columns:
            if col.find("foot") > -1:
                result_df[col] = round(result_df[col].apply(lambda x: 4878*(x if x>10 else df[col].mean())**(-0.837)),2)
        return result_df


class WeightCalculator(object):
    def __init__(self, col_list):
        self.__col_list = col_list

    def transform(self, df):
        result_df = df[self.__col_list["no_features"]]
        sw = np.ones_like(result_df)
        for wt in self.__col_list['weights']:
            sw = sw * df[wt].values  # 生成权重
        result_df.loc[:, "weight"] = sw
        return result_df.join(df[self.__col_list["features"]],how="inner")


# 解析文件夹
class DirParser(object):
    def __init__(self, raw_data_path=None):
        raw_data_path = r'E:/my_proj/fog_recognition/ExtendFoGData/raw_data' if not raw_data_path else raw_data_path
        self.__raw_data_path = path.Path(raw_data_path)

    def parse(self):
        # 初始化相关量
        file_df = pd.DataFrame()
        data_group, sample_times, signal_file, video_result_file,pid = [], [], [], [], []

        # 遍历路径获取数据全名称
        for fp in os.listdir(self.__raw_data_path):
            temp_fp = []
            try:
                file_pat = glob.glob(os.path.join(self.__raw_data_path, fp, "[a-z0-9A-Z_]*.xlsx"))
                temp_fp.extend([i.replace("\\", "/") for i in file_pat if i.find("result") == -1])
            except Exception as e:
                print(e)
            signal_file.extend(temp_fp)
            video_result_file.extend([os.path.join(self.__raw_data_path, fp, "result_1.xlsx").replace("\\", "/")] * len(temp_fp))

        c = 0
        for temp_fp in signal_file:
            c += 1
            (filepath, tempfilename) = os.path.split(temp_fp)
            (filename, extension) = os.path.splitext(tempfilename)
            data_group.append(os.path.basename(filepath))
            sample_times.append(filename)
            pid.append('pd{:0>6d}'.format(c))

        file_df['id'] = pid
        file_df["signal_file"] = signal_file
        file_df["video_result_file"] = video_result_file
        file_df["data_group"] = data_group
        file_df["sample_times"] = sample_times

        return file_df


class VideoMarkParser(object):
    sheet_structure = {"repeat_step": 5, "detail_data_start": (4, 2), "video_time_start": (1, 2),
                       "confidence_start": (3, 5), "simulation_effect_start": (37, range(3))}
    unit_map = {"video_result": pd.DataFrame(), "start_time": 0, "unit_confidence": 0, "video_len": 0}
    video_data_field = ["label_start_time", "label_end_time", "gait_type", "label_confidence", "stimulate_type"]

    def __init__(self):
        self.para = dict(sheet_structure=self.sheet_structure,
                         unit_map=self.unit_map,
                         video_data_field=self.video_data_field)

    def set_para_with_prop(self, my_prop):
        self.para.update(my_prop)

    def parse_sheet(self, sheet, times):
        sheet_structure = self.para['sheet_structure']
        start_col_no = (int(times) - 1) * sheet_structure["repeat_step"] + sheet_structure["detail_data_start"][1]
        end_col_no = start_col_no + sheet_structure["repeat_step"]
        start_row_no = sheet_structure["detail_data_start"][0]
        for i in range(500):
            try:
                if not str(sheet.iloc[i + start_row_no, start_col_no]).isdigit():
                    break
                else:
                    end_row_no = i + start_row_no
            except Exception as e:
                print(e)
        data = sheet.iloc[start_row_no:end_row_no + 1, start_col_no:end_col_no]
        data.columns = self.para['video_data_field']
        # 将时间转化为十进制绝对值
        row_no = sheet_structure["video_time_start"][0]
        col_no = (int(times) - 1) * sheet_structure["repeat_step"] + sheet_structure["video_time_start"][1]

        data["start_time"] = (data["label_start_time"]) * 100 + 0 + int(str(sheet.iloc[row_no, col_no]), 16)
        data["end_time"] = (data["label_end_time"]) * 100 + 99 + int(str(sheet.iloc[row_no, col_no]), 16)
        print("+++++++++++++++++++++++++++++++++")

        # 获取本次采用整体置信度
        row_no = sheet_structure["confidence_start"][0]
        col_no = (int(times) - 1) * sheet_structure["repeat_step"] + sheet_structure["confidence_start"][1]
        data["unit_confidence"] = sheet.iloc[row_no, col_no]
        return data

    def parse_vd_mark_workbook(self, workbook, request):
        sample_groups = request['sample_groups']
        samples = request['samples']
        sample_times = request['sample_times']
        res_names = request['res_names']
        result_book = pd.read_excel(workbook, sheet_name=list(samples[workbook]), header=None)
        c = 0
        res = {}
        for sample_time in sample_times[workbook]:
            sample_time_list = sample_time.split("_")
            result_df = VideoMarkParser().parse_sheet(result_book[sample_time_list[0]], sample_time_list[1])
            con = filter(lambda it: it.find(sample_groups[workbook]) >= 0 and it.find(sample_time) >= 0, res_names)
            res[next(con)] = result_df
        return res


STATUS_DEFINITION = {"fog":["A","B","C","D","F"],"normal":["H","I","J","K","N"]}
INFO_COLS = ["time10","label_start_time","label_end_time","gait_type",
             "label_confidence","stimulate_type","start_time","end_time","unit_confidence"]


class SignalMarker(object):
    """将这些关系紧密的代码最好不分开为妙"""
    @staticmethod
    def __check(data):
        print(data)

    @staticmethod
    def __mark_signal_with_video(df_s, df_v):
        df_all = df_s.reindex(columns=list(df_s.columns) + list(df_v.columns))
        for i in df_v.index:
            v_it = df_v.loc[i, :]
            start = v_it['start_time']
            end = v_it['end_time']
            idx = df_all[(df_all['time10'] >= start) & (df_all['time10'] <= end)].index.tolist()
            df_all.loc[idx, v_it.index.tolist()] = v_it.tolist()
        return df_all

    @staticmethod
    def __drop_head_tail_na(df_all, col_name):
        start_id = 0
        for i in df_all.index:
            if pd.notna(df_all.loc[i, col_name]):
                start_id = i
                break
        end_id = len(df_all)
        for j in df_all.index[::-1]:
            if pd.notna(df_all.loc[j, col_name]):
                end_id = j
                break
        print(start_id, end_id)
        return df_all.loc[start_id:end_id, :]

    @staticmethod
    def __mark_status(it, STATUS_DEFINITION, status="normal"):
        for k, lst in STATUS_DEFINITION.items():
            if it in lst:
                status = k
                break
        return status

    def calculate(self, msg):
        """the low layer(the general layer) should not know much about the above(the business)"""
        self.__check(msg)
        df_s = pd.read_csv(open(msg[0]))
        df_v = pd.read_csv(open(msg[1]))
        df_all = self.__mark_signal_with_video(df_s, df_v)
        df_res = self.__drop_head_tail_na(df_all, 'end_time')
        gait_type = list(itertools.chain.from_iterable(STATUS_DEFINITION.values()))
        # 将不再已知列表内的步态全部标记为N
        df_res.loc[:, 'gait_type'] = df_res['gait_type'].apply(lambda it: it if it in gait_type else 'N')
        # 根据步态类型所对应的状态，将status标记为fog与normal
        df_res.loc[:, 'status'] = df_res['gait_type'].apply(self.__mark_status, args=(STATUS_DEFINITION,))
        # 根据status,将其编码为数字
        df_res.loc[:, 'status_code'] = df_res.loc[:, 'status'].apply(lambda it: 1 if it == 'fog' else 0)
        # 根据状态码将其标记为独热码
        df_res.loc[:, 'one_hot'] = df_res.loc[:, 'status_code'].apply(keras.utils.to_categorical, args=(2,))

        self.__check(df_res['status_code'].value_counts())

        return df_res

