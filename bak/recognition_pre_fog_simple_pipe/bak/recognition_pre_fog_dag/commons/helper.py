"""
@Name:        helper
@Description: used to strengthen
@Author:      Lucas Yu
@Created:     2018/10/17
@Copyright:   (c) GYENNO Science,Shenzhen,Guangdong 2018
@Licence:
"""
import path


# 用于根据需求增强不同的Processor节点，而不是所有的节点的必须的
class ProcessorHelper(object):
    """加强节点功能"""
    @classmethod
    def find_item_in_container(cls, pat, container):
        res = None
        for it in container:
            if not path.Path(it).find(pat) < 0:
                res = it
                break
        return res

    @classmethod
    def get_node_input(cls, predecessors):
        pre_files = {predecessor.name: path.Path(predecessor.get_output_destination()).files() for predecessor in predecessors}
        # 求取ID的交集
        c = 0
        all_files_id_set = set()
        for predecessor_name, files in pre_files.items():
            c += 1
            tmp = set(map(lambda it: path.Path(it).basename().split('==')[0], files))
            if c == 1:
                all_files_id_set = all_files_id_set.union(tmp)
            else:
                all_files_id_set = all_files_id_set.intersection(tmp)

        data_batch = []
        for file_id in all_files_id_set:
            input_item = {}
            for predecessor_name, files in pre_files.items():
                input_item[predecessor_name] = cls.find_item_in_container(file_id, files)
            data_batch.append(input_item)
        return data_batch

    @classmethod
    def input_whose_output_exist(cls, processor, data_id):
        files = path.Path(processor.get_output_destination()).files()
        if cls.find_item_in_container(data_id, files):
            status = True
        else:
            status = False
        return status

    @classmethod
    def pair_input_by_id(cls, fnames, separator='=='):
        res = {}
        for node, v in fnames.items():
            res[node] = {it.basename().split(separator)[0]: it for it in v.files()}

        c = 0
        res_id = set()
        for node, v in res.items():
            c += 1
            if c == 1:
                res_id = set(v)
            res_id = res_id.intersection(set(v.keys()))

        res_li ={}
        for it in sorted(res_id):
            res_tmp = {}
            for node, v in res.items():
                res_tmp[node] = v.get(it)
            res_li[it] = res_tmp
        return res_li
