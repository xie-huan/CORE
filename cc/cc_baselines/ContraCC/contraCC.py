import os
import copy
import sys
# from read_data.ManyBugsDataLoader import ManyBugsDataLoader
# from read_data.Defects4JDataLoader import Defects4JDataLoader
# from read_data.SIRDataLoader import SIRDataLoader
from cc.ReadData import ReadData
from CONFIG import *

if __name__ == "__main__":

    configs = {'-d': 'd4j', '-p': 'Chart', '-i': '1', '-m': 'dstar', '-e': 'origin'}
    sys.argv = ["ReadData.py"]
    gpl = ReadData(project_dir, configs)
    gpl.load_data()
    a = 1