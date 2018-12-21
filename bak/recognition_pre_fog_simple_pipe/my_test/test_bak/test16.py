import joblib,path


class VideoMarkETL(object):
    def __init__(self, name=None, dependencies=None, reset=False):
        """实例化对象对象时必用"""
        self.request = {}

    def output(self, name_info=None, new_name=None):
        """输入和输出有关系"""
        info = [name_info['id'], "==", name_info['data_group'], "_", name_info['sample_times'],"_result_1.csv"]
        res_name = ''.join(info)
        return path.Path("").joinpath(res_name)

    def prepare(self):
        data = joblib.load('../data/raw_data')
        sample_groups, samples, sample_times, result_names = {}, {}, {}, {}
        for idx in data.index:
            sample_groups.update({data.loc[idx, 'video_result_file']: data.loc[idx, 'data_group']})
            result_names.update({self.output(name_info=data.loc[idx].to_dict()): data.loc[idx, 'video_result_file']})
            samples[data.loc[idx, 'video_result_file']] = set()
            sample_times[data.loc[idx, 'video_result_file']] = []
        for idx in data.index:
            samples[data.loc[idx, 'video_result_file']].add(data.loc[idx, 'sample_times'].split("_")[0])
            sample_times[data.loc[idx, 'video_result_file']].append(data.loc[idx, 'sample_times'])

        self.request['workbooks'] = list(sample_groups.keys())
        self.request['sample_groups'] = sample_groups
        self.request['samples'] = samples
        self.request['sample_times'] = sample_times
        self.request['result_names'] = result_names

        res_name_tmp = {res_name: mark_name for res_name, mark_name in result_names.items() if not path.Path(res_name).exists()}
        self.request['workbooks'] = set(res_name_tmp.values())
        for k,v in self.request.items():
            print(k,v)

from recognition_pre_fog.extract_transform_load import ExtractorMetaData



if __name__== "__main__":
    res= ExtractorMetaData().transform()

    for k in vars(res):
        print(k,'->', getattr(res,k))
    #VideoMarkETL().prepare()