from .metrics import *
from .dl_metrics import *
from .fs_metrics import *

def calc_corr(data, method):
    features_list = list(data.columns)[:-1]
    label = list(data.columns)[-1]
    corr_dict = {}

    if method == 'dstar':
        for feature in features_list:
            corr_dict[feature] = dstar(data[feature], data[label])
    elif method == 'barinel':
        for feature in features_list:
            corr_dict[feature] = barinel(data[feature], data[label])
    elif method == "ER1":
        for feature in features_list:
            corr_dict[feature] = ER1(data[feature], data[label])
    elif method == "ER2":
        for feature in features_list:
            corr_dict[feature] = ER2(data[feature], data[label])
    elif method == "ER3":
        for feature in features_list:
            corr_dict[feature] = ER3(data[feature], data[label])
    elif method == "ER4":
        for feature in features_list:
            corr_dict[feature] = ER4(data[feature], data[label])
    elif method == "ER5":
        for feature in features_list:
            corr_dict[feature] = ER5(data[feature], data[label])
    elif method == "ER6":
        for feature in features_list:
            corr_dict[feature] = ER6(data[feature], data[label])
    elif method == "Kulczynski2":
        for feature in features_list:
            corr_dict[feature] = Kulczynski2(data[feature], data[label])
    elif method == 'ochiai':
        for feature in features_list:
            corr_dict[feature] = ochiai(data[feature], data[label])
    elif method == "M2":
        for feature in features_list:
            corr_dict[feature] = M2(data[feature], data[label])
    elif method == "AMPLE2":
        for feature in features_list:
            corr_dict[feature] = AMPLE2(data[feature], data[label])
    elif method == "Wong3":
        for feature in features_list:
            corr_dict[feature] = Wong3(data[feature], data[label])
    elif method == "AM":
        for feature in features_list:
            corr_dict[feature] = AM(data[feature], data[label])
    elif method == "Cohen":
        for feature in features_list:
            corr_dict[feature] = Cohen(data[feature], data[label])
    elif method == "Fleiss":
        for feature in features_list:
            corr_dict[feature] = Fleiss(data[feature], data[label])
    elif method == "GP02":
        for feature in features_list:
            corr_dict[feature] = GP02(data[feature], data[label])
    elif method == "GP03":
        for feature in features_list:
            corr_dict[feature] = GP03(data[feature], data[label])
    elif method == "GP19":
        for feature in features_list:
            corr_dict[feature] = GP19(data[feature], data[label])
    # elif method == "ER2":
    #     for feature in features_list:
    #         corr_dict[feature] = ER2(data[feature], data[label])
    elif method == "Op2":
        for feature in features_list:
            corr_dict[feature] = Op2(data[feature], data[label])
    elif method == "Jaccard":
        for feature in features_list:
            corr_dict[feature] = Jaccard(data[feature],data[label])

    elif method == "MLP-FL":
        corr_dict = MLP(data[features_list], data[label])
    elif method == "CNN-FL":
        corr_dict = CNN(data[features_list], data[label])
    elif method == "RNN-FL":
        corr_dict = RNN(data[features_list], data[label])

    elif method == "pearson":
        for feature in features_list:
            corr_dict[feature] = pearson(data[feature], data[label])
    elif method == "spearman":
        for feature in features_list:
            corr_dict[feature] = spearman(data[feature], data[label])
    elif method == "kendall":
        for feature in features_list:
            kendall_corr = stats.kendalltau(data[feature].tolist(), data[label].tolist())
            corr_dict[feature] = round(kendall_corr[0], 6)
    elif method == "chisquare":
        for feature in features_list:
            corr_dict[feature] = chisq(data[feature], data[label])
    elif method == "mutual_information":
        for feature in features_list:
            corr_dict[feature] = NMI(data[feature], data[label])
    elif method == "fisher_score":
        sample = data.iloc[:, :-1].values.tolist()
        label = data.iloc[:, -1].values.tolist()

        fisher_score_list = binary_fisher_score(sample, label)
        corr_dict = dict(zip(features_list, fisher_score_list))
    else:
        raise Exception(f"Argument value error: No method '{method}'")
    return corr_dict
