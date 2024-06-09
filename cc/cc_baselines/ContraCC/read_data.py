import json

import numpy as np
import os
data_dir = "/Users/yuanxixing/Documents/study/master/2021-2022-2/1-CC/processed"

for program in os.listdir(data_dir):
    # if program != "Closure":
    #     continue

    program_dir = os.path.join(data_dir, program)
    if not os.path.isdir(program_dir):
        continue
    for version in os.listdir(program_dir):
        version_dir = os.path.join(program_dir, version)
        if not os.path.isdir(version_dir):
            continue
        matrix_dir = os.path.join(version_dir, "predict_full.json")
        if not os.path.exists(matrix_dir):
            continue


        with open(matrix_dir, "r") as f:
            a = json.load(f)
            precision = a["accuracy"]
            recall = a["recall"]
            f1 = a["f1"]
            with open("./result.txt", "a") as rf:
                print(program, version, precision, recall, f1, file=rf)
            # print(a)
        # a = json.loads(matrix_dir)
        # with open(matrix_dir, "r") as f:
        #     predict = f.load()

            # print(predict)
        # matrix = np.load(matrix_dir)
        a = 1
        # deal_dataset(ver_list, features, formulas, pro_path, coverage_file, origin_path, pro_name)