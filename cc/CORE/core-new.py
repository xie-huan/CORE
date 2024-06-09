import sys
import time

from CONFIG import *
from utils.postprocess import parse
from utils.write_util import write_rank_to_txt

from multiprocessing import Pool


# set number of CPUs to run on
ncore = "5"

# set env variables
# have to set these before importing numpy
os.environ["OMP_NUM_THREADS"] = ncore
os.environ["OPENBLAS_NUM_THREADS"] = ncore
os.environ["MKL_NUM_THREADS"] = ncore
os.environ["VECLIB_MAXIMUM_THREADS"] = ncore
os.environ["NUMEXPR_NUM_THREADS"] = ncore

import concurrent.futures


def run(program_list, start_program, start_program_id, identifyMethod, way, arg_dict):
    flag = False
    for program in program_list:
        for i in cc_info[program]:
            print(program, i)
            if program == start_program and i == start_program_id:
                flag = True
            if flag:
                configs = {'-d': 'd4j', '-p': program, '-i': i, '-m': method_para, '-e': 'origin'}
                # configs = {'-d': 'd4j', '-p': "Closure-2023-12-6-1", '-i': 36, '-m': method_para, '-e': 'origin'}
                sys.argv = os.path.basename(__file__)

                max_value, max_record, max_original_record, max_config = 0, None, None, None
                max_pl = None

                for n in arg_dict["cce_threshold"]:
                    model_arg = {"cce_threshold": n,
                                 "select_ratio":arg_dict["select_ratio"],
                                 "sus_threshold":arg_dict["sus_threshold"]}
                    pl = identifyMethod(project_dir, configs, model_arg, way)
                    start = time.time()
                    pl.find_cc_index()
                    end = time.time()
                    time_ = dict()
                    time_["time"] = end - start
                    save_path = os.path.join(project_dir, "new_results", way, "time.txt")
                    write_rank_to_txt(time_, save_path, program, i)

                    value, record, original_record, config = pl._out()
                    if value > max_value:
                        max_value, max_record, max_original_record, max_config = value, record, original_record, config
                        max_pl = pl
                if max_original_record is None or max_record is None or max_config is None:
                    continue
                original_record_path = os.path.join(project_dir, "new_results", way, "origin_record.txt")
                record_path = os.path.join(project_dir, "new_results", way, "record.txt")
                write_rank_to_txt(max_original_record, original_record_path, program, i)
                write_rank_to_txt(max_record, record_path, program, i)

                config_path = os.path.join(project_dir, "new_results", way, "config.txt")
                write_rank_to_txt(max_config, config_path, program, i)
            # break
                # pl.evaluation()
                max_pl.calRes("trim")
    parse(os.path.join(project_dir, "new_results", way), way+"_MFR.txt", "FL.xlsx")
    parse(os.path.join(project_dir, "new_results", way), "origin_record.txt", "precision_recall.xlsx")


from multiprocessing import Process

def multi_process_run(program_list, identifyMethod, way, arg_dict):
    p_list = []
    for program in program_list:
        p = Process(target=run, args=([program],program,cc_info[program][0],identifyMethod,way+program,arg_dict))
        p_list.append(p)
    [p.start() for p in p_list]
    [p.join() for p in p_list]
