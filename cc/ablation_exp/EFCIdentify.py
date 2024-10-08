import copy
import math
import random
import sys
import numpy as np
import pandas as pd
import torch

import torch.optim as optim

from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.MLP_DFL_cc_model.ReadTrainData import ReadTrainData
from cc.expert_feature_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.expert_feature_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.expert_feature_cc_identify.PassingTestsHandler import PassingTestsHandler
from cc.expert_feature_cc_model.EFCDataLoader import CombinedInfoLoader, CombinedInfoLoaderWithoutCovInfo
from cc.expert_feature_cc_model.ExpertFeatureCombinedNetwork import EFCNetwork, Net1, Net2, Net3, Net4, \
    CoverageInfoSematicNet, ExpertFeatureCombinedNetwork
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--combined_epochs', type=int, default=30, metavar='N',
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

args = parser.parse_args()


class EFCIdentify(BaseCCPipeline):
    def __init__(self, project_dir, configs, cita, way, K):
        super().__init__(project_dir, configs, way)
        self.cita = cita
        self.K = K
        self.train_test_config_list = []
        self.train_config_list = []
        self.test_config_list = []
        self.test_part_list = []
        self.ccpl4vote = {}
        self.target4vote = {}

    def _find_cc_index(self):
        program = self.configs['-p']
        info_list = cc_info[program]
        program_len = len(info_list)
        program_method = self.configs['-m']
        random.shuffle(info_list)
        weight = 1
        self.train_test_config_list.clear()

        for i in range(self.K):
            self.train_test_config_list.append([])
        for i in range(program_len):
            config = {'-d': 'd4j', '-p': program, '-i': str(info_list[i]), '-m': program_method,
                      '-e': 'origin'}
            self.train_test_config_list[i % self.K].append(config)

        for item in self.train_test_config_list:
            if len(item) == 0:
                continue
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
                ssp, cr, sf = FeatureTestsHandler.get_feature_from_file(project_dir, ccpl.program,
                                                                        ccpl.bug_id)
                ssp, cr, sf = ssp.values, cr.values, sf.values
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
            train_ssp = torch.vstack(tuple(train_ssp_list))
            train_cr = torch.vstack(tuple(train_cr_list))
            train_sf = torch.vstack(tuple(train_sf_list))
            train_cc_target = torch.vstack(train_cc_target_list)

            test_rtd = ReadTrainData(self.project_dir, item, self.way)
            self.test_part_list = test_rtd.ccpls

            train_loader = torch.utils.data.DataLoader(
                CombinedInfoLoaderWithoutCovInfo(target=train_cc_target,
                                                 ssp=train_ssp,
                                                 cr=train_cr,
                                                 sf=train_sf,
                                                 ),
                batch_size=min(args.batch_size, train_ssp.shape[0]),
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            # build model
            model1 = Net1(train_ssp.shape[1])
            model2 = Net2(train_ssp.shape[1])
            model3 = Net3(train_ssp.shape[1])
            model4 = Net4(train_ssp.shape[1] * 3)
            mnet = EFCNetwork(model1, model2, model3, model4)
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
                self._train_mnet(train_loader, mnet, criterion, optimizer, epoch)

            rtd = ReadTrainData(self.project_dir, item, self.way)
            train_list = rtd.ccpls
            # self._test_(model, test_rtd)
            for ccpl in train_list:
                CCE = find_CCE(ccpl)
                if len(CCE) == 0:
                    continue
                CCE.append("error")
                new_data_df = ccpl.data_df[CCE]

                ccpl.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)
                ccpl.passing_tests = PassingTestsHandler.get_passing_tests(new_data_df)

                size = ccpl.ground_truth_cc_index.shape[0]
                indices = ccpl.ground_truth_cc_index.index.to_numpy()
                np.random.shuffle(indices)

                indices = indices.tolist()

                if size < 5:
                    k = size
                else:
                    k = 5
                train_index_list = []
                for i in range(k):
                    train_index_list.append([])

                for i, item in enumerate(indices):
                    train_index_list[i % k].append(item)

                for i, test_index in enumerate(train_index_list):
                    train_index = []
                    for j, array in enumerate(train_index_list):
                        if j != i:
                            train_index += array
                    train_tests = ccpl.passing_tests.loc[train_index, :]
                    train_tests = train_tests.iloc[:, :-1]
                    ground_truth_train_target = ccpl.ground_truth_cc_index.loc[train_index]
                    # train_target = torch.FloatTensor(dataframe_target.loc[train_index].values)
                    train_target = torch.FloatTensor([[0, 1]] * len(train_index))
                    for i in range(len(train_index)):
                        if ground_truth_train_target[train_index[i]]:
                            train_target[i] = torch.FloatTensor([0, 1])
                        else:
                            train_target[i] = torch.FloatTensor([1, 0])

                    ccpl.ssp, ccpl.cr, ccpl.sf = FeatureTestsHandler.get_feature_from_file(project_dir, ccpl.program,
                                                                                           ccpl.bug_id)
                    ssp_feature = ccpl.ssp.loc[train_index, :]
                    cr_feature = ccpl.cr.loc[train_index, :]
                    sf_feature = ccpl.sf.loc[train_index, :]

                    train_loader = torch.utils.data.DataLoader(
                        CombinedInfoLoader(tests=train_tests * weight,
                                           target=train_target,
                                           ssp=ssp_feature,
                                           cr=cr_feature,
                                           sf=sf_feature
                                           ),
                        batch_size=min(args.batch_size, ccpl.passing_tests.shape[0]),
                        shuffle=True,
                        num_workers=0,
                        pin_memory=True,
                    )

                    criterion = torch.nn.MSELoss()

                    # coverage info model
                    elements_length = len(CCE) - 1
                    cover_info_net = CoverageInfoSematicNet(elements_length)
                    optimizer_cover_info = optim.SGD(cover_info_net.parameters(), lr=args.lr, momentum=args.momentum)
                    # optimizer_cover_info = optim.Adam(cover_info_net.parameters(), lr=args.lr)
                    if args.cuda:
                        cover_info_net.cuda()
                    for epoch in range(1, args.epochs):
                        self._train_cnet(train_loader, cover_info_net, criterion, optimizer_cover_info, epoch)

                    mnet_copy = copy.deepcopy(mnet)

                    model = ExpertFeatureCombinedNetwork(cover_info_net, mnet_copy)

                    if args.cuda:
                        model.cuda()

                    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

                    for epoch in range(1, args.combined_epochs):
                        self._train(train_loader, model, criterion, optimizer, epoch)
                    self._test(ccpl, model, test_index)
                key = ccpl.program + ccpl.bug_id
                if key not in self.target4vote:
                    self.target4vote[key] = []
                    self.ccpl4vote[key] = ccpl
                self.target4vote[key].append(ccpl.cc_index)

                # ccpl.evaluation()
                # ccpl.calRes("trim")

    def _train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (tests, target, ssp, cr, sf) in enumerate(train_loader):
            if args.cuda:
                tests, target, ssp, cr, sf = tests.cuda(), target.cuda(), ssp.cuda(), cr.cuda(), sf.cuda()
            tests, ssp, cr, sf, = tests.to(torch.float), ssp.to(torch.float), cr.to(torch.float), sf.to(torch.float)

            prob = model(tests, ssp, cr, sf)
            if args.cuda:
                target = target.cuda()
            loss = criterion(prob, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Train Epoch: {} [{}/{}]\t'
                      'loss: {}'.format(
                    epoch, batch_idx * len(target), len(train_loader.dataset),
                    loss,
                ))

    def _test(self, ccpl, model, test_index):
        model.eval()
        with torch.no_grad():
            for item in test_index:
                test = ccpl.passing_tests.loc[item, :]
                test = test.iloc[:-1]
                test, ssp, cr, sf = test, ccpl.ssp.loc[item], ccpl.cr.loc[item], \
                                    ccpl.sf.loc[item]

                test, ssp, cr, sf = torch.tensor(test.values), torch.tensor(ssp.values), torch.tensor(
                    cr.values), torch.tensor(sf.values)

                if args.cuda:
                    test, ssp, cr, sf = test.cuda(), ssp.cuda(), cr.cuda(), sf.cuda()
                test = torch.unsqueeze(test.to(torch.float), dim=0)
                ssp = torch.unsqueeze(ssp.to(torch.float), dim=0)
                cr = torch.unsqueeze(cr.to(torch.float), dim=0)
                sf = torch.unsqueeze(sf.to(torch.float), dim=0)

                prob = model(test, ssp, cr, sf)

                if prob[0][0] < prob[0][1]:
                    ccpl.cc_index[item] = True

    def _train_mnet(self, train_loader, mnet, criterion, optimizer, epoch):
        mnet.train()
        for batch_idx, (target, ssp, cr, sf) in enumerate(train_loader):

            if args.cuda:
                ssp, cr, sf = ssp.cuda(), cr.cuda(), sf.cuda()

            ssp = ssp.to(torch.float)
            cr = cr.to(torch.float)
            sf = sf.to(torch.float)

            # compute output
            prob = mnet(ssp, cr, sf)

            target = target.float().view(prob.shape)

            if args.cuda:
                target = target.cuda()

            loss = criterion(prob, target)

            # compute gradient and do optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Train mnet Epoch: {} [{}/{}]\t'
                      'loss: {}'.format(
                    epoch, batch_idx * len(ssp), len(train_loader.dataset),
                    loss,
                ))

    def _train_cnet(self, train_loader, cnet, criterion, optimizer, epoch):
        cnet.train()
        for batch_idx, (tests, target, _, _, _) in enumerate(train_loader):
            if args.cuda:
                tests, target = tests.cuda(), target.cuda()
            tests = tests.to(torch.float)
            prob = cnet(tests)

            if args.cuda:
                target = target.cuda()

            loss = criterion(prob, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print('Train CoverageInfo Epoch: {} [{}/{}]\t'
                      'loss: {}'.format(
                    epoch, batch_idx * len(target), len(train_loader.dataset),
                    loss,
                ))

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
    program_list = ["Chart", "Lang", "Math", "Mockito", "Time"]
    # program_list = ["Chart"]
    for program in program_list:
        configs = {'-d': 'd4j', '-p': program, '-i': '1', '-m': method_para, '-e': 'origin'}
        sys.argv = os.path.basename(__file__)
        cbccpl = EFCIdentify(project_dir, configs, 1, "2024-1-10-EFC-v2", 10)
        for i in range(2):
            cbccpl.find_cc_index()
        cbccpl.vote()
