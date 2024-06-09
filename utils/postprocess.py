import pandas as pd
from CONFIG import *


def parse(file_dir, input_file_name, output_file_name):
    data = pd.read_csv(os.path.join(file_dir, input_file_name), sep="\t", header=None)
    data = data.iloc[:, :-1]

    cc_list = []
    for program in program_list:
        for i in cc_info[program]:
            cc_list.append(program + "-" + str(i))

    bool_list = pd.Series([False] * data.shape[0])
    for index, row in data.iterrows():
        if row[0] in cc_list:
            bool_list[index] = True

    data_slice = data[bool_list]

    # excel_name = "FL.xlsx"
    excel_path = os.path.join(file_dir, output_file_name)

    if output_file_name == "precision_recall.xlsx":
        a = data_slice.iloc[:, 1:]
        a = a.sum(axis=0)
        precision = a.iloc[2] / a.iloc[1]
        recall = a.iloc[2] / a.iloc[0]
        F1 = 2 * precision * recall / (precision + recall)
        metric = pd.Series(["-", "recall", "precision", "F1"])
        data_slice = data_slice.append(metric, ignore_index=True)
        metric = pd.Series(["-", recall, precision, F1])
        data_slice = data_slice.append(metric, ignore_index=True)

    with pd.ExcelWriter(excel_path) as writer:
        data_slice.to_excel(writer, sheet_name="default")


def survey_parse():
    data_dir = os.path.join(project_dir, "new_results")

    all_data_precision_relabel = pd.DataFrame()
    all_data_precision_trim = pd.DataFrame()
    all_data_recall_relabel = pd.DataFrame()
    all_data_recall_trim = pd.DataFrame()

    for i in range(100, 9, -10):
        folder_name_precision_relabel = str(i) + "-precision-relabel-relabel"
        folder_name_precision_trim = str(i) + "-precision-trim-trim"
        folder_name_recall_relabel = str(i) + "-recall-relabel-relabel"
        folder_name_recall_trim = str(i) + "-recall-trim-trim"


        data_precision_relabel = pd.read_csv(os.path.join(data_dir, folder_name_precision_relabel, folder_name_precision_relabel+"_MFR.txt"), sep="\t", header=None)
        data_precision_relabel = data_precision_relabel.iloc[:, :-1]
        all_data_precision_relabel = pd.concat([all_data_precision_relabel, data_precision_relabel], axis=1)

        data_precision_trim = pd.read_csv(os.path.join(data_dir, folder_name_precision_trim, folder_name_precision_trim+"_MFR.txt"), sep="\t", header=None)
        data_precision_trim = data_precision_trim.iloc[:, :-1]
        all_data_precision_trim = pd.concat([all_data_precision_trim, data_precision_trim], axis=1)

        data_recall_relabel = pd.read_csv(os.path.join(data_dir, folder_name_recall_relabel, folder_name_recall_relabel+"_MFR.txt"), sep="\t", header=None)
        data_recall_relabel = data_recall_relabel.iloc[:, :-1]
        all_data_recall_relabel = pd.concat([all_data_recall_relabel, data_recall_relabel], axis=1)

        data_recall_trim = pd.read_csv(os.path.join(data_dir, folder_name_recall_trim, folder_name_recall_trim+"_MFR.txt"), sep="\t", header=None)
        data_recall_trim = data_recall_trim.iloc[:, :-1]
        all_data_recall_trim = pd.concat([all_data_recall_trim, data_recall_trim], axis=1)

    with pd.ExcelWriter("precision_relabel.xlsx") as writer:
        all_data_precision_relabel.to_excel(writer, sheet_name="default")

    with pd.ExcelWriter("precision_trim.xlsx") as writer:
        all_data_precision_trim.to_excel(writer, sheet_name="default")

    with pd.ExcelWriter("recall_relabel.xlsx") as writer:
        all_data_recall_relabel.to_excel(writer, sheet_name="default")

    with pd.ExcelWriter("recall_trim.xlsx") as writer:
        all_data_recall_trim.to_excel(writer, sheet_name="default")


    # parse(os.path.join(project_dir, "new_results", "survey"), "origin_record.txt", "precision_recall.xlsx")
