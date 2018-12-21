# ---------------------------------------------------------
# Name:        component to process
# Description: some fundamental component
# Author:      Lucas Yu
# Created:     2018-04-11
# Copyright:   (c) Zhenluo,Shenzhen,Guangdong 2018
# Licence:
# ---------------------------------------------------------
import path
import joblib
from concurrent import futures
from collections import deque

from ..client.proj_config import MyNode
from ..commons.common import MyProperties, MyProcessor

"""关注点分离,主要处理调度逻辑，是不打算去关系业务逻辑"""

class SignalETLInit(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)

    def output(self, name_info=None, new_name=None):
        return path.Path(self.get_output_destination()).joinpath(name_info)

    def prepare(self):
        print("accepted 消息：", self.request)
        if path.Path(self.output('raw_data')).exists():
            self.finished = True
            print(self.name, '的结果已经缓存，无需再次计算！')

    def process(self):
        from .algo_core import DirParser
        print("{0} has run !".format(self.name))
        raw_data_path = r'E:/my_proj/fog_recognition/ExtendFoGData/raw_data'
        dir_parser = DirParser(raw_data_path)
        joblib.dump(dir_parser.parse(), self.output('raw_data'))
        #fp = os.path.abspath('../recognition_pre_fog_dag/s1_etl_signal_video/config/raw_input.conf')
        #raw_input = ConfigFactory.parse_file(fp)
        #joblib.dump(raw_input, self.output(fp))
        #print(self.output(fp))


# 类似handler/controller
class SignalETL(MyProcessor):
    """
    responsible for scheduler logic and the do some initial job
    """
    def __init__(self, name=None, dependencies=None, reset=False):
        super(SignalETL, self).__init__(name, dependencies, reset)
        self.para.setdefault("SignalETLService", None)

    def output(self, name_info=None, new_name=None):
        result_name = name_info['id']+'=='+name_info['data_group']+name_info['sample_times']+'.csv'
        return path.Path(self.get_output_destination()).joinpath(result_name)

    def prepare(self):
        fp = path.Path(self.request[MyNode.SignalETLInit.name]).files()[0]
        data = joblib.load(fp, 'r').to_dict('records')
        self.request[MyNode.SignalETLInit.name] = [it for it in data if not self.output(it).exists()]
        if not self.request[MyNode.SignalETLInit.name]:
            self.finished = True
            print(self.name, '的结果已经缓存，无需再次计算！')

    def process(self):
        data = deque(self.request[MyNode.SignalETLInit.name])
        with futures.ProcessPoolExecutor(max_workers=6) as executor:
            future_to_it = {executor.submit(self.para["SignalETLService"].calculate, it): it for it in data}
            for future in futures.as_completed(future_to_it):
                it = future_to_it[future]
                print('future_to_it', it)
                future.result().to_csv(self.output(it), index=False)


class VideoMarkETL(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        """实例化对象对象时必用"""
        super(VideoMarkETL, self).__init__(name, dependencies, reset)

    def output(self, name_info=None, new_name=None):
        """输入和输出有关系"""
        info = [name_info['id'], "==", name_info['data_group'], "_", name_info['sample_times'],"_result_1.csv"]
        res_name = ''.join(info)
        return path.Path(self.get_output_destination()).joinpath(res_name)

    def prepare(self):
        data = joblib.load(path.Path(self.request[MyNode.SignalETLInit.name]).files()[0], 'r')
        sample_groups, samples, sample_times, res_names = {}, {}, {}, {}
        for idx in data.index:
            sample_groups.update({data.loc[idx, 'video_result_file']: data.loc[idx, 'data_group']})
            res_names.update({self.output(name_info=data.loc[idx].to_dict()): data.loc[idx, 'video_result_file']})
            samples[data.loc[idx, 'video_result_file']] = set()
            sample_times[data.loc[idx, 'video_result_file']] = []
        for idx in data.index:
            samples[data.loc[idx, 'video_result_file']].add(data.loc[idx, 'sample_times'].split("_")[0])
            sample_times[data.loc[idx, 'video_result_file']].append(data.loc[idx, 'sample_times'])

        self.request['workbooks'] = list(sample_groups.keys())
        self.request['sample_groups'] = sample_groups
        self.request['samples'] = samples
        self.request['sample_times'] = sample_times
        self.request['res_names'] = res_names

        res_name_tmp = {res_name: mark_name for res_name, mark_name in res_names.items() if not path.Path(res_name).exists()}
        self.request['workbooks'] = set(res_name_tmp.values())

        if not self.request['workbooks']:
            self.finished = True
            print(self.name, '的结果已经缓存，无需再次计算！')

    def process(self):
        from .algo_core import VideoMarkParser
        for itm in self.request['workbooks']:
            par = VideoMarkParser()
            res = par.parse_vd_mark_workbook(itm, self.request)
            for k, v in res.items():
                con = filter(lambda it: it.find(k) >= 0, list(self.request['res_names'].keys()))
                res[k].to_csv(next(con), index=False)


class SignalMark(MyProcessor):
    def __init__(self, name=None, dependencies=None, reset=False):
        super().__init__(name, dependencies, reset)

    def set_para_with_prop(self, my_props):
        self.para.setdefault("SignalMarker", None)
        self.para.update(my_props)

    def output(self, name_info=None, new_name=None):  # 动态构建本任务的输出结果的名称，因为其值取决于输入参数
        return path.Path(self.get_output_destination()).joinpath(path.Path(name_info).basename())

    def prepare(self):
        # 相当于输入函数input_fn，准备数据之外，还可以做些其他工作
        # 获取输入的名称，并根据输入名称算出输出名称，然后结合输出是否存在修改输入
        files_name = {node: path.Path(data_url).files() for node, data_url in self.request.items()}
        # 若对应输入的输出不存在，则将该输入记录，用于将来计算
        processing_ids = [it.basename().split('==')[0] for it in files_name[MyNode.SignalETL.name] if not self.output(name_info=it).exists()]

        data = []
        for idx in processing_ids:
            res = {}
            for node, names in files_name.items():
                tmp = list(filter(lambda it: it.find(idx) > -1, names))
                if len(tmp) > 0:
                    res[node] = tmp[0]
            if len(res.keys()) == len(files_name.keys()):
                data.append(res)

        if not data:
            self.finished = True
            print(self.name, '的结果已经缓存，无需再次计算！')
        self.request = data

    def process(self):
        data = deque(self.request)
        with futures.ProcessPoolExecutor(max_workers=self.para['max_workers']) as executor:
            future_to_it = {executor.submit(self.para["SignalMarker"].calculate,
                                            [it[MyNode.SignalETL.name], it[MyNode.VideoMarkETL.name]]): it for it in data}
            for future in futures.as_completed(future_to_it):
                it = future_to_it[future]
                future.result().to_csv(self.output(name_info=it[MyNode.SignalETL.name]), index=False)










