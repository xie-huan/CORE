import os

import numpy as np
import pandas as pd

from cc.MLP_DFL_cc_identify.Features.Features import Features
from sklearn.preprocessing import StandardScaler


class FeatureTestsHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_feature_from_file(project_dir, program, bug_id):
        save_path = os.path.join(project_dir, "feature", "MLP", f"{program}-passing-csv-1")
        # file_path = f"{save_path}/cce-{cita}-{program}-{bug_id}.npy"
        file_path = f"{save_path}/features-{program}-{bug_id}.csv"

        feature_matrix = pd.read_csv(file_path, index_col=0)

        # loaded_matrix = np.load(file_path)
        ssp = feature_matrix.iloc[:, 0:10]
        cr = feature_matrix.iloc[:, 10:20]
        sf = feature_matrix.iloc[:, 20:30]

        ssp_array = ssp.to_numpy()
        cr_array = cr.to_numpy()
        sf_array = sf.to_numpy()

        standardScaler1 = StandardScaler()
        standardScaler2 = StandardScaler()
        standardScaler3 = StandardScaler()
        standardScaler1.fit(ssp_array)
        standardScaler2.fit(cr_array)
        standardScaler3.fit(sf_array)
        ssp_standard = standardScaler1.transform(ssp_array)
        cr_standard = standardScaler2.transform(cr_array)
        sf_standard = standardScaler3.transform(sf_array)

        ssp = pd.DataFrame(ssp_standard, index=ssp.index)
        sf = pd.DataFrame(sf_standard, index=sf.index)
        cr = pd.DataFrame(cr_standard, index=cr.index)
        return ssp, cr, sf
