import numpy as np
import torch.utils.data as data


class MLPDFLTestCaseLoader(data.Dataset):

    def __init__(self, susScore, covRatio, similarityFactor, cc_target):
        self.susScore_np = np.array(susScore)
        self.covRatio_np=np.array(covRatio)
        self.similarityFactor_np=np.array(similarityFactor)
        self.cc_target=cc_target

    def __getitem__(self, item):
        return self.susScore_np[item],self.covRatio_np[item],self.similarityFactor_np[item],self.cc_target[item]

    def __len__(self):
        return self.susScore_np.shape[0]