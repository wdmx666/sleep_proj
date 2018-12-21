"""
@Name:        services
@Description: ''
@Author:      Lucas Yu
@Created:     2018/11/4
@Copyright:   ©GYENNO Technology,Shenzhen,Guangdong 2018
@Licence:
"""
import collections
import path
import pandas as pd
from ..commons.common import MyCalculator


class EventParaValidator(MyCalculator):
    def __init__(self, name=None):
        super().__init__(name)
        self.para.setdefault("strategy", None)
        self.para.setdefault("data_path", None)
        self.para.setdefault("save_path", None)

    def set_para_with_prop(self, my_props):
        self.para.update(my_props)

    def calculate(self, msg):
        c = 0
        ID, para = msg
        que = collections.deque(path.Path(self.para['data_path']).files())
        all_predicted_time = dict()
        rs = pd.DataFrame()
        print("que：", len(que))
        while que:
            item = que.popleft()
            df_tmp = pd.read_csv(open(item))
            mark_result = self.para['strategy'].mark_result(df_tmp, para["filter_time"], para["probability_value"],para["event_time"])
            res_all = Utils.evaluation(mark_result)
            print(res_all)
            res = res_all[0]
            predicted_time = res_all[1]
            f1 = metrics.f1_score(res["truth"], res["predict"], average="macro")
            c += 1
            res["sample_name"] = path.Path(item).basename()
            rs = rs.append(res)
            all_predicted_time.update({path.Path(item).basename(): predicted_time})
            print(rs.columns,"=============================================the %d" % c)

        #print(all_predicted_time)
        save_path = path.Path(self.para["save_path"]).joinpath("event_para_"+ID)
        #input_data = all_predicted_time.values()
        #print(metrics.classification_report(rs["truth"], rs["predict"]))
        print(rs.head())
        #print(metrics.f1_score(rs["truth"], rs["predict"], average="macro"), metrics.cohen_kappa_score(rs["truth"], rs["predict"]))
        EventPlotter().calculate((save_path, [[j for j in i if j is not None] for i in all_predicted_time.values()]))  # 偷了一个懒，没有注入
        return all_predicted_time, rs