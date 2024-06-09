import numpy as np

class Evaluation():
    def __init__(self):
        pass

    @staticmethod
    def evaluation(ground_truth_cc_index, cc_index):
        real_cc_num = np.sum(np.array(ground_truth_cc_index, dtype=int))
        ours_cc_num = np.sum(np.array(cc_index, dtype=int))
        correct_series = ground_truth_cc_index & cc_index
        cc_intersection_num = np.sum(np.array(correct_series, dtype=int))
        original_record = dict()
        original_record["real_cc_num"] = real_cc_num
        original_record["ours_cc_num"] = ours_cc_num
        original_record["cc_intersection_num"] = cc_intersection_num

        record = dict()

        if real_cc_num == 0:

            if ours_cc_num == 0:
                record["recall"] = 1
                record["precision"] = 1
                record["F1"] = 1
            else:
                record["recall"] = 0
                record["precision"] = 0
                record["F1"] = 0
        else:
            if ours_cc_num == 0:
                record["recall"] = 0
                record["precision"] = 0
                record["F1"] = 0
            else:
                # calculate precision recall and F1
                # recall
                recall = cc_intersection_num / real_cc_num
                # print("recall", str(recall))
                # precision
                precision = cc_intersection_num / ours_cc_num
                # print("precision", str(precision))
                # F1
                if recall == 0 or precision == 0:
                    F1 = 0
                else:
                    F1 = 2 * precision * recall / (precision + recall)

                record["recall"] = recall
                record["precision"] = precision
                record["F1"] = F1
        return original_record, record
