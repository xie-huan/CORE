import sys
import pandas as pd

from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from CONFIG import *


class ReadTrainData:
    def __init__(self, project_dir, configs,way):
        self.ccpls=[]
        for config in configs:
            ccpl=BaseCCPipeline(project_dir, config, way)
            ccpl.init_cc_index()
            self.ccpls.append(ccpl)



if __name__=="__main__":
    configs1 = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    configs2 = {'-d': 'd4j', '-p': 'Chart', '-i': '1', '-m': 'dstar', '-e': 'origin'}
    configs=[configs1,configs2]
    rld=ReadTrainData(project_dir,configs)
    print(rld.ccpls[0].cc_index,rld.ccpls[1].cc_index)