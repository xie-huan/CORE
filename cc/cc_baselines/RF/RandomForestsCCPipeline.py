import copy
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.core import run
from utils.task_util import task_complete


class RandomForestsCCPipeline(BaseCCPipeline):
    def __init__(self, project_dir, configs, N, way):
        super().__init__(project_dir, configs, way)
        self.fCCE = []
        self.N = N

    # def find_cc_index(self):
    #     data_df = self.data_obj.data_df
    #     if len(data_df[data_df["error"] == 0]) == 0:
    #         record = dict()
    #         recourd["msg"] = "No passing tests"
    #         save_path = os.path.join(self.project_dir, "results", self.way, "record.txt")
    #         write_rank_to_txt(record, save_path, self.program, self.bug_id)
    #         return
    #     self._find_cc_index()

    def _find_cc_index(self):
        data_df = self.load_data()
        PT_data = data_df[data_df["error"] == 0]
        PT_data_copy = copy.deepcopy(PT_data)
        PT_len = len(PT_data_copy)
        FT_data = data_df[data_df["error"] == 1]
        # a = math.floor(PT_data.shape[0]*0.8)

        K = max(int(len(PT_data) * 0.2), 1)  # 0.8 is selectable, and it repesents the percent of train test
        self.N = min(self.N, PT_data.shape[0] - K)
        if self.N == 0:
            return
        PT_data = PT_data.sample(frac=1)
        CT = []
        for i in range(0, len(PT_data), K):
            PT_split = self.PTsetSplit(pd.concat([PT_data.iloc[:i, :], PT_data.iloc[i + K:, :]]), self.N)
            PT_Test = PT_data.iloc[i:i + K, :]
            train = []
            RF_classifier = []
            Zm = []
            for j in range(self.N):
                Zm.append([])
            for j in range(self.N):
                train.append(pd.concat([PT_split[j], FT_data]))
                train_set = train[j].iloc[:, [m for m in range(PT_data.shape[1] - 1)]]
                train_label = train[j].iloc[:, [PT_data.shape[1] - 1]]
                test = PT_Test
                test_set = test.iloc[:, [m for m in range(PT_data.shape[1] - 1)]]
                test_label = test.iloc[:, [PT_data.shape[1] - 1]]
                RF = RandomForestClassifier(n_estimators=10, random_state=0)
                RF.fit(train_set, np.array(train_label.T)[0])
                RF_classifier.append(RF)
                for item in test_set.index:
                    loc_item = test_set.index.get_loc(item)
                    Zm[j].append(RF.score(test_set.iloc[loc_item, :].to_frame().values.reshape(-1, 1).T,
                                          test_label.iloc[loc_item, :].to_frame().values.reshape(-1, 1).T))
            for j in range(len(PT_Test)):
                cc_num = 0
                ncc_num = 0
                for Z in Zm:
                    if Z[j] == 1:
                        cc_num += 1
                    else:
                        ncc_num += 1
                if cc_num >= ncc_num:
                    CT.append(PT_Test.index.values.tolist()[j])
        CT_Bool = pd.DataFrame([False for i in range(PT_len)], index=PT_data_copy.index.values.tolist())
        CT_Bool.loc[CT, :] = True
        final_CT = CT_Bool.values.T[0]
        self.cc_index = pd.Series(final_CT, index=self.ground_truth_cc_index.index)

    # pt is the set to be split， and n is the number of piece after split，result represent the result after split
    def PTsetSplit(self, pt, n):
        shuffled = pt.sample(frac=1)
        result = np.array_split(shuffled, n)
        return result


def main():
    program_list = [
        "Chart",
        "Closure-2023-12-6-1",
        "Lang",
        "Math",
        "Mockito",
        "Time"
    ]
    run(program_list, "Chart", 1, RandomForestsCCPipeline, "2024-RF", 5)



if __name__ == "__main__":
    main()
    task_complete("RF end")
    # configs = {'-d': 'd4j', '-p': "Math", '-i': 62, '-m': method_para, '-e': 'origin'}
    # sys.argv = os.path.basename(__file__)
    # svmccpl = RandomForestsCCPipeline(project_dir, configs, 5, "RandomForest")
    # svmccpl.find_cc_index()
    # svmccpl.evaluation()
    # svmccpl.calRes()
