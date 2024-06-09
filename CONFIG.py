import os
import warnings

from pandas.core.common import SettingWithCopyWarning

os.environ["OMP_NUM_THREADS"] = '1'
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
project_dir = os.path.dirname(__file__)

method_list = [
    # "ER1",
    # "ER2",
    # "ER3",
    # "ER4",
    # "ER5",
    # "ER6",
    # "Kulczynski2",
    # "ochiai",
    # "M2",
    # "AMPLE2",
    # "Wong3",
    # "AM",
    # "Cohen",
    # "Fleiss",
    "ER1",
    "ER5",
    "GP02",
    "GP03",
    "GP19",
    "dstar",
    "ochiai",
    "barinel",
    # "Op2"
    # "MLP-FL",
    # "CNN-FL",
    # "RNN-FL"
    ]
method_para = ""
for method in method_list[:-1]:
    method_para += method + ","
method_para += method_list[-1]

program_list = [
    "Chart",
    "Lang",
    "Math",
    "Mockito",
    "Time"
]
method_para = ""
for method in method_list[:-1]:
    method_para += method + ","
method_para += method_list[-1]

program_version_num_list = [
    26,
    133,
    65,
    106,
    38,
    27
]

import yaml

cc_info = None
with open(os.path.join(project_dir,"cc","CCinfo.yaml"), 'r') as stream:
    try:
        cc_info = yaml.safe_load(stream)
    except Exception as exc:
        print(exc)
#
# print(cc_info)

all_info = None
with open(os.path.join(project_dir, "cc", "allinfo.yaml"), 'r') as stream:
    try:
        all_info = yaml.safe_load(stream)
    except Exception as exc:
        print(exc)


test_info = None
with open(os.path.join(project_dir, "cc", "testinfo.yaml"), 'r') as stream:
    try:
        test_info = yaml.safe_load(stream)
    except Exception as exc:
        print(exc)
