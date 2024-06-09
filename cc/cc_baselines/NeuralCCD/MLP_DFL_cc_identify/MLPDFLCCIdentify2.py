import math
import random
import sys
import time

import pandas as pd
import torch

import torch.optim as optim

from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.MLP_DFL_cc_identify.FeatureTestHandler import FeatureTestsHandler
from cc.MLP_DFL_cc_model.MLPDFLTestCaseLoader import MLPDFLTestCaseLoader
from cc.MLP_DFL_cc_model.MLPDFLNet import Net1, Net2, Net3, Net4, MLPDFLnet
from cc.MLP_DFL_cc_model.ReadTrainData import ReadTrainData
from utils.write_util import write_rank_to_txt
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

class CBNNCCIdentify2(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way, K):
        super().__init__(project_dir, configs, way)
        self.traindata = []
        self.testdata = []
        self.cita = cita
        self.K = K
        self.train_test_config_list = []
        self.train_config_list = []
        self.test_config_list = []
        self.test_part_list = []
        self.ccpl4vote = {}
        self.target4vote = {}
        self.cc_target = None

    def _find_cc_index(self):
        program = self.configs['-p']
        info_list = cc_info[program]
        program_len = len(info_list)
        program_method = self.configs['-m']
        random.shuffle(info_list)
        self.train_test_config_list.clear()

        for i in range(self.K):
            self.train_test_config_list.append([])

        for i in range(program_len):
            config = {'-d': 'd4j', '-p': program, '-i': str(info_list[i]), '-m': program_method,
                      '-e': 'origin'}
            self.train_test_config_list[i % self.K].append(config)

        start = time.time()
        for item in self.train_test_config_list:
            train_list = self.train_test_config_list.copy()
            train_list.remove(item)
            train_list = [i for item in train_list for i in item]
            train_rtd = ReadTrainData(self.project_dir, train_list, self.way)
            train_CCE_list = []
            train_ssp_list = []
            train_cr_list = []
            train_sf_list = []
            train_cc_target_list = []
            for ccpl in train_rtd.ccpls:
                CCE = find_CCE(ccpl)
                train_CCE_list.append(CCE)
                if len(CCE) == 0:
                    continue
                CCE.append("error")
                # new_data_df = ccpl.data_df[CCE]
                # 使用文件的方式导入ssp,cr,sf特征矩阵信息
                ssp, cr, sf = FeatureTestsHandler.get_feature_from_file(project_dir, self.cita, ccpl.program,
                                                                        ccpl.bug_id)
                train_ssp_list.append(torch.FloatTensor(ssp))
                train_cr_list.append(torch.FloatTensor(cr))
                train_sf_list.append(torch.FloatTensor(sf))
                cc_target = torch.FloatTensor([[0, 1]] * ssp.shape[0])
                target = ccpl.ground_truth_cc_index.astype("int").values
                for i in range(len(target)):
                    if target[i] == 1:
                        cc_target[i] = torch.FloatTensor([0, 1])
                    else:
                        cc_target[i] = torch.FloatTensor([1, 0])
                train_cc_target_list.append(cc_target)
                # print(ccpl.program+ccpl.bug_id+" end")

            train_ssp = torch.vstack(tuple(train_ssp_list))
            train_cr = torch.vstack(tuple(train_cr_list))
            train_sf = torch.vstack(tuple(train_sf_list))
            train_cc_target = torch.vstack(train_cc_target_list)

            test_rtd = ReadTrainData(self.project_dir, item, self.way)
            self.test_part_list = test_rtd.ccpls

            train_loader = torch.utils.data.DataLoader(
                MLPDFLTestCaseLoader(susScore=train_ssp,
                                     covRatio=train_cr,
                                     similarityFactor=train_sf,
                                     cc_target=train_cc_target
                                     ),
                batch_size=min(args.batch_size, ssp.shape[0]),
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # build model
            model1 = Net1(train_ssp.shape[1])
            model2 = Net2(train_ssp.shape[1])
            model3 = Net3(train_ssp.shape[1])
            model4 = Net4(train_ssp.shape[1] * 3)
            mnet = MLPDFLnet(model1, model2, model3, model4)
            if args.cuda:
                mnet.cuda()

            # loss function and optimizer
            # criterion = torch.nn.MarginRankingLoss(margin=args.margin)
            # criterion = torch.nn.CrossEntropyLoss()
            criterion = torch.nn.MSELoss()
            # criterion=torch.nn.NLLLoss()
            optimizer = optim.Adam(mnet.parameters(), lr=args.lr)
            # optimizer = optim.SGD(mnet.parameters(), lr=args.lr, momentum=args.momentum)

            # train the model
            # EPOCH = 100
            for epoch in range(1, args.epochs):
                # self.train(train_loader, mnet, criterion, optimizer, cc_target, epoch)
                self.train(train_loader, mnet, criterion, optimizer, epoch)
            end = time.time()
            time_ = dict()
            time_["time"] = end - start
            save_path = os.path.join(project_dir, "results", "2024-1-5-CBNN-30", "time.txt")
            write_rank_to_txt(time_, save_path, program, self.train_test_config_list.index(item))
            torch.save(mnet.state_dict(), program + "_" + str(self.train_test_config_list.index(item)) + '_mnet.pt')

            self._test_(mnet, test_rtd)
            # for ccpl in self.test_part_list:
            #     ccpl.evaluation()
            #     ccpl.calRes("trim")

    def train(self, train_loader, mnet, criterion, optimizer, epoch):
        mnet.train()
        for batch_idx, (susScore, covRatio, similarityFactor, target) in enumerate(train_loader):

            if args.cuda:
                susScore, covRatio, similarityFactor = susScore.cuda(), covRatio.cuda(), similarityFactor.cuda()

            susScore = susScore.to(torch.float)
            covRatio = covRatio.to(torch.float)
            similarityFactor = similarityFactor.to(torch.float)

            # compute output
            # Research question: random strategy or vote?
            prob = mnet(susScore, covRatio, similarityFactor)

            # print(prob)

            # target = target[:, 1]
            # target = torch.from_numpy(target.values)
            # 将张量组织成prob.shape形状的的浮点型序列
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

    def _test_(self, mnet, test_rtd):
        for ccpl in test_rtd.ccpls:
            CCE = find_CCE(ccpl)
            if len(CCE) == 0:
                return
            CCE.append("error")
            # new_data_df = ccpl.data_df[CCE]
            ssp, cr, sf = FeatureTestsHandler.get_feature_from_file(project_dir, self.cita, ccpl.program, ccpl.bug_id)
            susScore_features = pd.DataFrame(ssp, index=ccpl.cc_index.index.values)
            covRatio_features = pd.DataFrame(cr, index=ccpl.cc_index.index.values)
            similarityFactor_features = pd.DataFrame(sf, index=ccpl.cc_index.index.values)

            mnet.eval()
            with torch.no_grad():
                for index, susScore_feature in susScore_features.iterrows():
                    susScore = torch.tensor(susScore_feature.values)
                    covRatio = torch.tensor(covRatio_features.loc[index].values)
                    similarityFactor = torch.tensor(similarityFactor_features.loc[index].values)

                    if args.cuda:
                        susScore, covRatio, similarityFactor = susScore.cuda(), covRatio.cuda(), similarityFactor.cuda()

                    # ssp = susScore.to(torch.float)
                    # cr = covRatio.to(torch.float)
                    # sf = similarityFactor.to(torch.float)

                    ssp = torch.unsqueeze(susScore.to(torch.float), dim=0)
                    cr = torch.unsqueeze(covRatio.to(torch.float), dim=0)
                    sf = torch.unsqueeze(similarityFactor.to(torch.float), dim=0)

                    prob = mnet(ssp, cr, sf)
                    prob = prob.view(1, 2)
                    if prob[0][0] < prob[0][1]:
                        ccpl.cc_index[index] = True
            key = ccpl.program + ccpl.bug_id
            if key not in self.target4vote:
                self.target4vote[key] = []
                self.ccpl4vote[key] = ccpl
            self.target4vote[key].append(ccpl.cc_index)

    def vote(self):
        for key in self.target4vote:
            ccpl = self.ccpl4vote[key]
            cc_index = pd.Series([False] * len(ccpl.ground_truth_cc_index.index),
                                 index=ccpl.ground_truth_cc_index.index)

            vote_list = self.target4vote[key]
            if len(vote_list) == 0:
                break
            for index in vote_list[0].index:
                CC_num = 0
                nCC_num = 0
                for j in range(len(vote_list)):
                    if vote_list[j].loc[index]:
                        CC_num += 1
                    else:
                        nCC_num += 1
                if CC_num >= nCC_num:
                    cc_index.loc[index] = True
            ccpl.cc_index = cc_index
            ccpl.evaluation()
            ccpl.calRes("relabel")


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
    uncover = sum(data == 0)
    cover = sum(data == 1)
    pT = cover / (uncover + cover)
    return pT


if __name__ == "__main__":
    program_list = ["Chart"]
    # program_list = ["Chart", "Lang", "Math", "Mockito", "Time"]
    for program in program_list:
        configs = {'-d': 'd4j', '-p': program, '-i': '1', '-m': method_para, '-e': 'origin'}
        sys.argv = os.path.basename(__file__)
        cbccpl = CBNNCCIdentify2(project_dir, configs, 1, "2024-1-9-CBNN", 5)
        for i in range(5):
            cbccpl.find_cc_index()
        cbccpl.vote()
