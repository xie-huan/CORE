import os
import copy
import sys
from read_data.ManyBugsDataLoader import ManyBugsDataLoader
from read_data.Defects4JDataLoader import Defects4JDataLoader
from read_data.SIRDataLoader import SIRDataLoader
from CONFIG import *

class ReadData:
    def __init__(self, project_dir, configs):
        self.configs = configs
        # self.project_dir = project_dir
        self.project_dir = "/Users/yuanxixing/Documents/study/master/2020-2021-2/1-ICSE2022/DATA_ENHANCE"
        self.dataset = configs["-d"]
        self.program = configs["-p"]
        self.bug_id = configs["-i"]
        self.method = configs["-m"].split(",")
        self.dataloader = self._choose_dataloader_obj()
        self.data_obj = copy.deepcopy(self.dataloader)

        self.data_df = self.load_data()

    def load_data(self):
        self.data_obj = copy.deepcopy(self.dataloader)
        return self.data_obj.data_df

    def _dynamic_choose(self, loader):
        self.dataset_dir = os.path.join(self.project_dir, "..")
        data_obj = loader(self.dataset_dir, self.program, self.bug_id)
        data_obj.load()
        return data_obj

    def _choose_dataloader_obj(self):
        if self.dataset == "d4j":
            return self._dynamic_choose(Defects4JDataLoader)
        if self.dataset == "manybugs" or self.dataset == "motivation":
            return self._dynamic_choose(ManyBugsDataLoader)
        if self.dataset == "SIR":
            return self._dynamic_choose(SIRDataLoader)

if __name__ == "__main__":

    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    sys.argv = ["ReadData.py"]
    gpl = ReadData(project_dir, configs)
    gpl.load_data()
    a = 1
