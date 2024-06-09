import numpy as np
from torch.utils import data


class CombinedInfoLoader(data.Dataset):
    def __init__(self, tests, target, ssp, cr, sf):
        self.tests = np.array(tests)
        self.target = np.array(target)
        self.ssp = np.array(ssp)
        self.cr = np.array(cr)
        self.sf = np.array(sf)

    def __getitem__(self, item):
        return self.tests[item], self.target[item], self.ssp[item], self.cr[item], self.sf[item]

    def __len__(self):
        return self.target.shape[0]


class CombinedInfoLoaderWithoutCovInfo(data.Dataset):
    def __init__(self, target, ssp, cr, sf):
        self.target = np.array(target)
        self.ssp = np.array(ssp)
        self.cr = np.array(cr)
        self.sf = np.array(sf)

    def __getitem__(self, item):
        return self.target[item], self.ssp[item], self.cr[item], self.sf[item]

    def __len__(self):
        return self.target.shape[0]


class CombinedInfoLoaderWithoutExpertFeature(data.Dataset):
    def __init__(self, tests, target):
        self.target = np.array(target)
        self.tests = np.array(tests)

    def __getitem__(self, item):
        return self.tests[item], self.target[item]

    def __len__(self):
        return self.target.shape[0]
