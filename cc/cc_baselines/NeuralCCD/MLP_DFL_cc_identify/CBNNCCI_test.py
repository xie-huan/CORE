import math
import sys
import random
import time

import numpy as np
import pandas as pd
import torch

import torch.optim as optim

from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.MLP_DFL_cc_identify.FeatureTestHandler import FeatureTestsHandler
from cc.MLP_DFL_cc_model.MLPDFLTestCaseLoader import MLPDFLTestCaseLoader
from cc.MLP_DFL_cc_model.MLPDFLNet import Net1,Net2,Net3,Net4,MLPDFLnet
from cc.MLP_DFL_cc_model.ReadTrainData import ReadTrainData
from utils.task_util import task_complete
from utils.write_util import write_rank_to_txt
from cc.triplet_cc_identify.FailingTestsHandler import FailingTestsHandler
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--cuda', type=bool, default=True, help='CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='MLPDFLCCNet', type=str,
                    help='name of experiment')

args = parser.parse_args()

# def setup_seed(seed):
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.deterministic = True
#
# # 设置随机数种子
# setup_seed(20)

class CBNNCCIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way,K):
        super().__init__(project_dir, configs, way)
        self.traindata=[]
        self.testdata=[]
        self.cita = cita
        self.K=K
        self.train_test_config_list = []
        self.train_config_list=[]
        self.test_config_list = []
        self.test_part_list=[]
        self.ssp=None
        self.cr=None
        self.sf=None
        self.cc_target = None


    def _find_cc_index(self):
        program=self.configs['-p']
        info_list=cc_info[program]
        # info_list=[62,63,64,65]
        program_len=len(info_list)
        program_method=self.configs['-m']
        # random.shuffle(info_list)
        K_size=round(program_len/self.K)
        test_list=[]
        for i in info_list:
            config = {'-d': 'd4j', '-p': program, '-i': str(i), '-m': program_method,
                      '-e': 'origin'}
            test_list.append(config)
        test_rtd = ReadTrainData(self.project_dir, test_list,self.way)
        self.test_part_list=test_rtd.ccpls
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        programs=["Chart","Math","Mockito","Lang","Time"]
        # programs = ["Mockito"]
        programs.remove(program)
        mnet_list=[]
        for program in programs:
            model1 = Net1(8)
            model2 = Net2(8)
            model3 = Net3(8)
            model4 = Net4(24)
            mnet = MLPDFLnet(model1, model2, model3, model4)
            mnet.load_state_dict(torch.load(program+"_test8_mnet.pt"))
            mnet = mnet.to(device)
            mnet_list.append(mnet)
        self._test_(mnet_list,test_rtd)
        for ccpl in self.test_part_list:
            ccpl.evaluation()
            ccpl.calRes("trim")


    def train(self,train_loader, mnet, criterion, optimizer ,epoch):
        # switch to train mode
        mnet.train()
        for batch_idx, (susScore,covRatio,similarityFactor,target) in enumerate(train_loader):

            if args.cuda:
                susScore,covRatio,similarityFactor = susScore.cuda(), covRatio.cuda(), similarityFactor.cuda()

            susScore = susScore.to(torch.float)
            covRatio = covRatio.to(torch.float)
            similarityFactor = similarityFactor.to(torch.float)

            # compute output
            # Research question: random strategy or vote?
            prob = mnet(susScore,covRatio,similarityFactor)

            # print(prob)

            # target = torch.from_numpy(target.values)
            target = target.float().view(prob.shape)

            if args.cuda:
                target = target.cuda()

            # loss_MLPDFL = criterion(prob, target.squeeze(dim=1).long())
            loss_MLPDFL = criterion(prob, target)

            loss = loss_MLPDFL

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if epoch % 10 == 0:
            #     print('Train Epoch: {} [{}/{}]\t'
            #           'loss: {}'.format(
            #         epoch, batch_idx * len(susScore), len(train_loader.dataset),
            #         loss,
            #     ))

            print('Train Epoch: {} [{}/{}]\t'
                  'loss: {}'.format(
                epoch, batch_idx * len(susScore), len(train_loader.dataset),
                loss,
            ))

    def _test_(self,mnet_list,test_rtd):
        for ccpl in test_rtd.ccpls:
            # print(test_rtd.ccpls.index(ccpl))
            CCE = find_CCE(ccpl)
            if len(CCE) == 0:
                new_data_df = ccpl.data_df
            else:
                CCE.append("error")
                new_data_df = ccpl.data_df[CCE]
            ssp, cr, sf = FeatureTestsHandler.get_feature_tests(new_data_df)
            susScore_features=pd.DataFrame(ssp,index=ccpl.cc_index.index.values)
            covRatio_features=pd.DataFrame(cr,index=ccpl.cc_index.index.values)
            similarityFactor_features=pd.DataFrame(sf,index=ccpl.cc_index.index.values)

            for mnet in mnet_list:
                mnet.eval()
            with torch.no_grad():
                for index, susScore_feature in susScore_features.iterrows():
                    susScore = torch.tensor(susScore_feature.values)
                    covRatio = torch.tensor(covRatio_features.loc[index].values)
                    similarityFactor = torch.tensor(similarityFactor_features.loc[index].values)

                    if args.cuda:
                         susScore, covRatio, similarityFactor= susScore.cuda(), covRatio.cuda(), similarityFactor.cuda()

                    ssp=susScore.to(torch.float)
                    cr=covRatio.to(torch.float)
                    sf=similarityFactor.to(torch.float)

                    prob_list=[]
                    for mnet in mnet_list:
                        prob = mnet(ssp,cr,sf)
                        prob = prob.view(1, 2)
                        prob_list.append(prob)

                    CC_num=0
                    nCC_num=0
                    for prob in prob_list:
                        if prob[0][1]>=0.5:
                            CC_num+=1
                        else:
                            nCC_num+=1
                    if CC_num>=nCC_num:
                        ccpl.cc_index[index]=True



    def getfT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        fT = cover / (uncover + cover)
        return fT

    def getpT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        pT = cover / (uncover + cover)
        return pT

    def _is_CCE(self, fail_data, pass_data):
        fT = self.getfT(fail_data)
        pT = self.getpT(pass_data)
        if fT == 1.0 and pT < self.cita:
            return True
        else:
            return False

    def _find_CCE(self):
        failing_df = self.data_df[self.data_df["error"] == 1]
        passing_df = self.data_df[self.data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i]):
                    CCE.append(i)
        self.CCE = CCE

def find_CCE(ccpl):
    failing_df = ccpl.data_df[ccpl.data_df["error"] == 1]
    passing_df = ccpl.data_df[ccpl.data_df["error"] == 0]
    CCE = []
    for i in failing_df.columns:
        if i != "error":
            if _is_CCE(failing_df[i], passing_df[i]):
                CCE.append(i)
    return CCE

def _is_CCE(fail_data, pass_data):
        fT = getfT(fail_data)
        pT = getpT(pass_data)
        if fT == 1.0 and pT < 1:
            return True
        else:
            return False

def getfT(data):
    uncover = sum(data == 0)
    cover = sum(data == 1)
    fT = cover / (uncover + cover)
    return fT

def getpT(data):
    if len(data)==0:
        return 1
    uncover = sum(data == 0)
    cover = sum(data == 1)
    pT = cover / (uncover + cover)
    return pT

if __name__ == "__main__":
    program_list=["Mockito"]
    for program in program_list:
        configs = {'-d': 'd4j', '-p': program, '-i': '1', '-m': method_para, '-e': 'origin'}
        sys.argv = os.path.basename(__file__)
        cbccpl = CBNNCCIdentify(project_dir, configs,1,"2022-10-08-CBNN-8",5)
        cbccpl.find_cc_index()
    a = 1
