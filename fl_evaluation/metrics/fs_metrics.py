import numpy as np
import pandas as pd
from scipy import stats
from sklearn import metrics
from .metrics import get_N_para


# Pearson 相关系数的实现，给定特征和标签series，返回相关系数值
def pearson(feature, label):
    # 自己实现的pearson算法
    cov = feature.cov(label)  # 协方差
    std_x = feature.std()  # 标准差
    std_y = label.std()  # 标准差
    if abs(std_x * std_y) < 1e-5:
        return np.nan
    else:
        pearson_corr = cov / (std_x * std_y)  # 特征与标签的相关系数
        # pearson_corr = min(1., max(-1., pearson_corr))  # 限制结果范围区间为[-1, 1]
        return pearson_corr

    # 调用stats实现的pearson
    # try:
    #     return stats.pearsonr(feature, label)[0]
    # except:
    #     return np.nan

def spearman(feature, label):
    # 排名没有并列的情况
    # feature_rank = data[feature].rank()
    # label_rank = data[label].rank()
    # diff = feature_rank.sub(label_rank)
    # diff_square = diff.mul(diff)
    #
    # N = int(data[label].count())
    # spearman_corr = 1 - 6 * sum(diff_square) / (N * (N * N - 1))
    # print(feature + ":" + str(spearman_corr))

    # 排名有并列的情况
    feature_rank = feature.rank()
    label_rank = label.rank()

    cov = feature_rank.cov(label_rank)  # 协方差
    std_x = feature_rank.std()  # 标准差
    std_y = label_rank.std()  # 标准差
    # 分母为0，相关系数则为nan
    if abs(std_x * std_y) < 1e-5:
        return np.nan
    else:
        spearman_corr = cov / (std_x * std_y)  # 特征与标签的相关系数
        # spearman_corr = min(1., max(-1., spearman_corr))  # 限制结果范围区间为[-1, 1]
        return spearman_corr

    # 调用stats实现的spearman
    # try:
    #     return stats.spearmanr(feature, label)[0]
    # except:
    #     return np.nan


# 计算kendall相关系数接口
# Kendall 相关系数的实现，给定特征和标签series，返回相关系数值
def kendall(feature, label):
    x = np.array(feature)
    y = np.array(label)

    size = x.size
    perm = np.argsort(y)
    x, y = x[perm], y[perm]
    y = np.r_[True, y[1:] != y[:-1]].cumsum(dtype=np.intp)

    perm = np.argsort(x, kind='mergesort')
    x, y = x[perm], y[perm]
    x = np.r_[True, x[1:] != x[:-1]].cumsum(dtype=np.intp)

    dis = calc_dis(y)  # discordant pairs

    obs = np.r_[True, (x[1:] != x[:-1]) | (y[1:] != y[:-1]), True]
    cnt = np.diff(np.nonzero(obs)[0]).astype('int64', copy=False)

    ntie = (cnt * (cnt - 1) // 2).sum()
    xtie = count_tie(x)
    ytie = count_tie(y)

    tot = (size * (size - 1)) // 2

    # tot = con + dis + (xtie - ntie) + (ytie - ntie) + ntie
    #     = con + dis + xtie + ytie - ntie
    con_minus_dis = tot - xtie - ytie + ntie - 2 * dis
    tau = con_minus_dis / np.sqrt(tot - xtie) / np.sqrt(tot - ytie)

    # tau = min(1., max(-1., tau))
    return tau


# 计算kendall相关系数中的xtie，ytie
def count_tie(vector):
    cnt = np.bincount(vector).astype('int64', copy=False)
    cnt = cnt[cnt > 1]
    return (cnt * (cnt - 1) // 2).sum()


# 求逆序对
def mergeSortInversion(data, aux, low, high):
    if low >= high:
        return 0

    mid = low + (high - low) // 2
    # 递归调用过程
    leftCount = mergeSortInversion(data, aux, low, mid)
    rightCount = mergeSortInversion(data, aux, mid + 1, high)

    # merge 过程
    for index in range(low, high + 1):
        aux[index] = data[index]
    count = 0
    i = low
    j = mid + 1
    k = i
    while k <= high:
        if i > mid and j <= high:
            data[k] = aux[j]
            j += 1
        elif j > high and i <= mid:
            data[k] = aux[i]
            i += 1
        elif aux[i] <= aux[j]:
            data[k] = aux[i]
            i += 1
        elif aux[i] > aux[j]:
            data[k] = aux[j]
            j += 1
            count += mid - i + 1
        k += 1

    return leftCount + rightCount + count

# 计算kendall相关系数中的不一致对
def calc_dis(y):
    aux = [y[i] for i in range(len(y))]
    nSwap = mergeSortInversion(y, aux, 0, len(y) - 1)
    return nSwap

# @xiehuan实现的卡方检验
def chisq(feature, label):
    # 实际值
    obs = list(get_N_para(feature, label))

    # 计算理论值
    fail_ratio = np.sum(label) / len(label)
    success_ratio = 1 - fail_ratio

    cover = np.sum(feature)
    uncover = len(feature) - cover

    Ncf = cover * fail_ratio
    Nuf = uncover * fail_ratio
    Ncs = cover * success_ratio
    Nus = uncover * success_ratio
    exp = [Ncf, Nuf, Ncs, Nus]

    return 1 - stats.chisquare(obs, exp)[1]

# @xiehuan实现的互信息（直接调包）
def NMI(feature, label):
    return metrics.normalized_mutual_info_score(feature, label)

# 标签只能为0和1,样本空间任意
# samle:n*m 的列表
# label:m*1 的列表
# 返回每个特征的fisher score值的一个列表
# eg:  sample = [[1,2,3],[1,0,1],[1,5,6]]
#     label = [1, 0, 1]
# return  lst=[nan, 1.8148148148148149, 1.8148148148148149]
def binary_fisher_score(sample, label):
    if len(sample) != len(label):
        print('Sample does not match label')
        exit()
    df1 = pd.DataFrame(sample)
    df2 = pd.DataFrame(label, columns=['label'])
    data = pd.concat([df1, df2], axis=1)  # 合并成为一个dataframe

    data0 = data[data.label == 0]  # 对标签分类，分成包含0和1的两个dataframe
    data1 = data[data.label == 1]
    n = len(label)  # 标签长度
    n1 = sum(label)  # 1类标签的个数
    n0 = n - n1  # 0类标签的个数
    lst = []  # 用于返回的列表
    features_list = list(data.columns)[:-1]
    for feature in features_list:

        # 算关于data0
        m0_feature_mean = data0[feature].mean()  # 0类标签在第m维上的均值
        # 0类在第m维上的sw
        m0_SW = sum((data0[feature] - m0_feature_mean) ** 2)
        # 算关于data1
        m1_feature_mean = data1[feature].mean()  # 1类标签在第m维上的均值
        # 1类在第m维上的sw
        m1_SW = sum((data1[feature] - m1_feature_mean) ** 2)
        # 算关于data
        m_all_feature_mean = data[feature].mean()  # 所有类标签在第m维上的均值

        m0_SB = n0 / n * (m0_feature_mean - m_all_feature_mean) ** 2
        m1_SB = n1 / n * (m1_feature_mean - m_all_feature_mean) ** 2
        # 计算SB
        m_SB = m1_SB + m0_SB
        # 计算SW
        m_SW = (m0_SW + m1_SW) / n
        if m_SW == 0:
            # 0/0类型也是返回nan
            m_fisher_score = np.nan
        else:
            # 计算Fisher score
            m_fisher_score = m_SB / m_SW
        # Fisher score值添加进列表
        lst.append(m_fisher_score)

    return lst
