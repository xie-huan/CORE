import json
import sys

import pandas as pd

from cc.CCGroundTruthPipeline import CCGroundTruthPipeline
from CONFIG import *
from cc.cc_evaluation.Evaluation import Evaluation
from fl_evaluation.calculate_suspiciousness.CalculateSuspiciousness import CalculateSuspiciousness
from utils.write_util import write_rank_to_txt


class BaseCCPipeline(CCGroundTruthPipeline):
    def __init__(self, project_dir, configs, way):
        super().__init__(project_dir, configs)
        self.cc_index = None
        self.way = way
        self.data_df = self.load_data()
        self.init_cc_index()

    def init_cc_index(self):
        self.cc_index = pd.Series([False] * len(self.ground_truth_cc_index.index),
                                  index=self.ground_truth_cc_index.index)

    def find_cc_index(self):

        if len(self.data_df[self.data_df["error"] == 0]) == 0:
            record = dict()
            record["msg"] = "No passing tests"
            save_path = os.path.join(self.project_dir, "new_results", self.way, "record.txt")
            write_rank_to_txt(record, save_path, self.program, self.bug_id)
            return

        self._find_cc_index()

    def evaluation(self):
        if self.cc_index is None:
            print("Calculate CC index first")
            return
        else:
            original_record, record = Evaluation.evaluation(self.ground_truth_cc_index, self.cc_index)
            original_record_path = os.path.join(self.project_dir, "new_results", self.way, "origin_record.txt")
            write_rank_to_txt(original_record, original_record_path, self.program, self.bug_id)
            record_path = os.path.join(self.project_dir, "new_results", self.way, "record.txt")
            write_rank_to_txt(record, record_path, self.program, self.bug_id)

    def calRes(self, operation):
        if self.cc_index is None:
            print("Calculate CC index first")
            return
        data_df = self.load_data()
        passing_df = data_df[data_df["error"] == 0]

        if operation == "relabel":
            passing_df["error"][self.cc_index] = 1
        if operation == "trim":
            passing_df = passing_df[self.cc_index == False]

        failing_df = data_df[data_df["error"] == 1]
        cc_data_df = pd.concat([passing_df, failing_df])
        self.data_obj.reload(cc_data_df)

        op_way = self.way+"-"+operation
        # op_way = self.way
        save_rank_path = os.path.join(self.project_dir, "new_results", op_way)
        cc = CalculateSuspiciousness(self.data_obj, self.method, save_rank_path, op_way)
        cc.run()


if __name__ == "__main__":

    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '26', '-m': 'dstar', '-e': 'origin'}
    sys.argv = os.path.basename(__file__)
    bpl = BaseCCPipeline(project_dir, configs, "test")
    a = bpl.ground_truth_cc_index[bpl.ground_truth_cc_index == True]

    a = 1

    # Contra_CC
    # data_dir = "/Users/yuanxixing/Documents/study/master/2021-2022-2/1-CC/processed"
    #
    # for program, versions in cc_info.items():
    #     if program != "Closure":
    #         continue
    #     program_dir = os.path.join(data_dir, program)
    #     for version in versions:
    #
    #         configs = {'-d': 'd4j', '-p': program, '-i': str(version), '-m': 'dstar', '-e': 'muse'}
    #         sys.argv = os.path.basename(__file__)
    #         bpl = BaseCCPipeline(project_dir, configs, "test")
    #         # a = bpl.ground_truth_cc_index[bpl.ground_truth_cc_index == True]
    #         try:
    #             version_dir = os.path.join(program_dir, str(version))
    #             if not os.path.isdir(version_dir):
    #                 continue
    #             matrix_dir = os.path.join(version_dir, "predict_full.json")
    #             if os.path.exists(matrix_dir):
    #                 with open(matrix_dir, "r") as f:
    #                     prediction = json.load(f)["predictions"]
    #             data_df = bpl.data_df
    #             data_df['error'] = prediction
    #             bpl.data_obj.reload(data_df)
    #         except Exception as e:
    #             print(e)
    #         try:
    #             bpl.calRes("default-new")
    #         except Exception as e:
    #             print(e)

    # Contra_CC
    # import yaml
    #
    # cc_info = None
    # with open(os.path.join(project_dir, "cc", "bugs.yaml"), 'r') as stream:
    #     try:
    #         cc_info = yaml.safe_load(stream)
    #     except Exception as exc:
    #         print(exc)
    #
    # for program, version in cc_info.items():
    #     if program == "Closure" or program == "Cli" or program == "Chart":
    #         continue
    #     for bug_id in version:
    #         try:
    #             configs = {'-d': 'd4j', '-p': program, '-i': bug_id, '-m': method_para, '-e': 'origin'}
    #             sys.argv = os.path.basename(__file__)
    #             bpl = BaseCCPipeline(project_dir, configs, "test")
    #             a = bpl.ground_truth_cc_index[bpl.ground_truth_cc_index == True]
    #             all_passing = len(bpl.ground_truth_cc_index)
    #             cc_tests = len(a)
    #             non_cc_tests = all_passing - cc_tests
    #             failing = len(bpl.data_df[bpl.data_df["error"] == 1])
    #             with open("status.txt", "a") as f:
    #                 f.write(program+"\t"+str(bug_id)+"\t"+str(all_passing)+"\t"+str(cc_tests)+"\t"+str(non_cc_tests)+"\t"+str(failing)+"\n")
    #         except Exception as e:
    #             continue