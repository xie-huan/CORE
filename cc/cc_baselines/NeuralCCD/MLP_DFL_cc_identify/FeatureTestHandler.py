import os

import numpy as np
import pandas as pd

from cc.MLP_DFL_cc_identify.Features.Features import Features
from sklearn.preprocessing import StandardScaler


class FeatureTestsHandler:
    def __init__(self):
        pass

    # 标准化处理
    @staticmethod
    def get_feature_tests(data_df):
        features = Features(data_df)
        ssp = features.suspScore()
        cr = features.covRatio()
        sf = features.similarityFactor()
        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        standardScaler3 = StandardScaler()
        standardScaler1.fit(ssp)
        standardScaler2.fit(cr)
        standardScaler3.fit(sf)
        ssp_standard = standardScaler1.transform(ssp)
        cr_standard = standardScaler2.transform(cr)
        sf_standard = standardScaler3.transform(sf)

        return ssp_standard, cr_standard, sf_standard

    @staticmethod
    def get_feature_from_file(project_dir, cita, program, bug_id):
        save_path = os.path.join(project_dir, "feature", "MLP", f"{program}-passing-csv-1")
        # file_path = f"{save_path}/cce-{cita}-{program}-{bug_id}.npy"
        file_path = f"{save_path}/features-{program}-{bug_id}.csv"

        # loaded_matrix = np.load(file_path)
        loaded_matrix = pd.read_csv(file_path, index_col=0)
        loaded_matrix=loaded_matrix.to_numpy()
        ssp = loaded_matrix[:, 0:10]
        cr = loaded_matrix[:, 10:20]
        sf = loaded_matrix[:, 20:30]
        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        standardScaler3 = StandardScaler()
        standardScaler1.fit(ssp)
        standardScaler2.fit(cr)
        standardScaler3.fit(sf)
        ssp_standard = standardScaler1.transform(ssp)
        cr_standard = standardScaler2.transform(cr)
        sf_standard = standardScaler3.transform(sf)

        return ssp_standard, cr_standard, sf_standard
