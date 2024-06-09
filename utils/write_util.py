import os
import pandas as pd
import yaml

def write_corr_to_txt(method, corr_dict, path, state):
    corr_dict = pd.DataFrame.from_dict(corr_dict, orient='index', columns=['susp'], dtype=float)
    corr_dict = corr_dict.reset_index().rename(columns={'index': 'line_num'})
    corr_dict['line_num'] = corr_dict['line_num'].astype(int)
    corr_dict.sort_values(by=['susp', 'line_num'], ascending=[False, True], inplace=True)

    save_path = os.path.join(path, state)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    res_file_name = method + "-" + state + ".txt"
    concrete_path = os.path.join(save_path, res_file_name)

    with open(concrete_path, 'w') as f:
        for each in corr_dict.values:
            print(str(int(each[0])) + "  " + str(each[1]), file=f)


def write_rank_to_txt(rank_dict, sava_path, program, bug_id):
    dir = os.path.dirname(sava_path)
    if not os.path.exists(dir):
        os.mkdir(dir)
    with open(sava_path, 'a') as f:
        value = f"{program}-{bug_id}\t"
        # for v in rank_dict.keys():
        #     value += str(v) + "\t"
        for v in rank_dict.values():
            value += str(v) + "\t"
        print(value, file=f)


def write_to_yaml(path: str, data: dict):
    with open(path, "w") as file:
        file.write(yaml.dump(data, allow_unicode=True))