import copy
import os
import sys
import numpy as np
import pandas as pd

from utils.args_util import parse_args
from CONFIG import *
from cc.ReadData import ReadData


class CCGroundTruthPipeline(ReadData):
    def __init__(self, project_dir, configs):
        super().__init__(project_dir, configs)
        self.ground_truth_cc_index = None
        # self.ground_truth_cc_index = np.zeros(self.data_obj.passing_df)
        # data_df = self.data_obj.data_df
        # fault_line = self.data_obj.fault_line

        # 筛选出成功测试用例, and init all_cc_index
        passing_df = self.data_df[self.data_df["error"] == 0]
        self.ground_truth_cc_index = pd.Series(np.array(np.zeros(len(passing_df)), dtype=bool), index=passing_df.index)

        self.load_ground_truth()
        self.ground_truth_cc_index_backup = copy.deepcopy(self.ground_truth_cc_index)


    def load_ground_truth(self):
        self._find_all_cc_index()

    # ground truth
    def _find_all_cc_index(self):
        data = self.data_obj
        data_df = data.data_df
        fault_line = data.fault_line
        # 筛选出成功测试用例
        passing_df = data_df[data_df["error"] == 0]
        # 找到偶然正确性测试用例，经过了有缺陷代码行的所有标定为正确的测试用例
        all_line = np.array(data.feature_df.columns)
        for line in fault_line:
            if line not in all_line:
                continue
            cc_index = passing_df[line] >= 1
            self.ground_truth_cc_index = self.ground_truth_cc_index | cc_index
        # self.ground_truth_cc_index = set(all_cc_index)
        # a = 1


if __name__ == "__main__":

    # project_dir = os.path.join(os.path.dirname(__file__), "..", "..")
    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    sys.argv = os.path.basename(__file__)
    ccpl = CCGroundTruthPipeline(project_dir, configs)
    ccpl.load_ground_truth()
    a = 1
