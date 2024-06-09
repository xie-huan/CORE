import math
import numpy as np
import torch
from torch import optim

from CONFIG import *
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.expert_feature_cc_identify.FailingTestsHandler import FailingTestsHandler
from cc.expert_feature_cc_identify.FeatureTestsHandler import FeatureTestsHandler
from cc.expert_feature_cc_identify.PassingTestsHandler import PassingTestsHandler
from cc.expert_feature_cc_model.EFCDataLoader import CombinedInfoLoaderWithoutCovInfo
import argparse

from cc.expert_feature_cc_model.ExpertFeatureCombinedNetwork import CoverageInfoSematicNet, Net1, Net2, Net3, Net4, \
    EFCNetworkWithoutCovInfo

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Training settings
parser = argparse.ArgumentParser(description='Triplet for CC')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train (default: 30)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
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
parser.add_argument('--name', default='TripletNet', type=str,
                    help='name of experiment')

args = parser.parse_args()

weight = 1


class EFCIdentifyWithoutCovInfo(BaseCCPipeline):
    def __init__(self, project_dir, configs, args_dict, way):
        super().__init__(project_dir, configs, way)
        self.CCT = None
        self.CCE = None
        self.feature = None
        self.args_dict = args_dict
        self.cita = None
        self.true_passing_tests = None
        self.failing_tests = None
        self.sus_dict = {}
        self.train_flag = True

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            self.train_flag = False
            return
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]

        self.failing_tests = FailingTestsHandler.get_failing_tests(new_data_df)
        self.passing_tests = PassingTestsHandler.get_passing_tests(new_data_df)

        target = self.ground_truth_cc_index.astype("int").values
        self.cc_target = torch.FloatTensor([[0, 1]] * self.passing_tests.shape[0])
        for i in range(len(target)):
            if target[i] == 1:
                self.cc_target[i] = torch.FloatTensor([0, 1])
            else:
                self.cc_target[i] = torch.FloatTensor([1, 0])

        size = self.ground_truth_cc_index.shape[0]
        indices = np.arange(size)
        np.random.shuffle(indices)

        k = 5
        part_size = math.ceil(size / k)

        # 依次取出
        for i in range(k):
            start = i * part_size
            end = (i + 1) * part_size if i < k - 1 else size
            test_index = indices[start:end]
            train_index = np.concatenate([indices[:start], indices[end:]])
            train_target = self.cc_target[train_index]

            self.ssp, self.cr, self.sf = FeatureTestsHandler.get_feature_from_file(project_dir, self.program,
                                                                                   self.bug_id)
            ssp_feature = self.ssp.iloc[train_index, :]
            cr_feature = self.cr.iloc[train_index, :]
            sf_feature = self.sf.iloc[train_index, :]

            train_loader = torch.utils.data.DataLoader(
                CombinedInfoLoaderWithoutCovInfo(
                    target=train_target,
                    ssp=ssp_feature,
                    cr=cr_feature,
                    sf=sf_feature
                ),
                batch_size=min(args.batch_size, self.passing_tests.shape[0]),
                shuffle=True,
                num_workers=0,
                pin_memory=True,
            )

            ssp_net = Net1(ssp_feature.shape[1])
            cr_net = Net2(cr_feature.shape[1])
            sf_net = Net3(sf_feature.shape[1])

            size = ssp_feature.shape[1] * 3
            combined_net = Net4(ssp_feature.shape[1] * 3)
            model = EFCNetworkWithoutCovInfo(ssp_net, cr_net, sf_net, combined_net, size)
            if args.cuda:
                model.cuda()
            # loss function and optimizer
            criterion = torch.nn.MSELoss()
            optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

            for epoch in range(1, args.epochs):
                self._train(train_loader, model, criterion, optimizer, epoch)
            self._test(model, test_index)

    def _train(self, train_loader, model, criterion, optimizer, epoch):
        model.train()
        for batch_idx, (target, ssp, cr, sf) in enumerate(train_loader):
            if args.cuda:
                target, ssp, cr, sf = target.cuda(), ssp.cuda(), cr.cuda(), sf.cuda()
            ssp = ssp.to(torch.float)
            cr = cr.to(torch.float)
            sf = sf.to(torch.float)
            prob = model(ssp, cr, sf)

            if args.cuda:
                target = target.cuda()

            loss = criterion(prob, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if epoch % 10 == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'loss: {}'.format(
                epoch, batch_idx * len(target), len(train_loader.dataset),
                loss,
            ))

    def _test(self, model, test_index):
        model.eval()
        with torch.no_grad():
            for item in test_index:
                ssp = self.ssp.iloc[item]
                cr = self.cr.iloc[item]
                sf = self.sf.iloc[item]

                ssp = torch.tensor(ssp.values)
                cr = torch.tensor(cr.values)
                sf = torch.tensor(sf.values)

                if args.cuda:
                    ssp, cr, sf = ssp.cuda(), cr.cuda(), sf.cuda()
                ssp = torch.unsqueeze(ssp.to(torch.float), dim=0)
                cr = torch.unsqueeze(cr.to(torch.float), dim=0)
                sf = torch.unsqueeze(sf.to(torch.float), dim=0)

                prob = model(ssp, cr, sf)

                if prob[0][0] < prob[0][1]:
                    self.cc_index[item] = True

    def _getfT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        fT = cover / (uncover + cover)
        return fT

    def _getpT(self, data):
        uncover = sum(data == 0)
        cover = sum(data == 1)
        pT = cover / (uncover + cover)
        return pT

    def _is_CCE(self, fail_data, pass_data, cita):
        fT = self._getfT(fail_data)
        pT = self._getpT(pass_data)
        if fT == 1.0 and pT < cita:
            return True
        else:
            return False

    def _find_CCE(self):
        if "cce_threshold" not in self.args_dict:
            column = self.data_df.columns[:-1]
            self.CCE = list(column)
            return
        self.cita = self.args_dict["cce_threshold"]
        failing_df = self.data_df[self.data_df["error"] == 1]
        passing_df = self.data_df[self.data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i], self.cita):
                    CCE.append(i)
        self.CCE = CCE

    def get_TP_when_already_find_cce(self, data_df, feature_matrix):
        passing_df = data_df[data_df["error"] == 0]
        new_data_df = passing_df.drop(passing_df.columns[-1], axis=1)
        sum_df = new_data_df.sum(axis=1)

        cc_candidate_list = list(sum_df[sum_df > 0].index)
        true_passing_list = list(sum_df[sum_df == 0].index)
        true_passing_test = data_df.iloc[true_passing_list, :].astype('float32')
        cc_candidate = data_df.iloc[cc_candidate_list, :].astype('float32')
        true_passing_test_feature = feature_matrix.loc[true_passing_list]
        cc_candidate_feature = feature_matrix.loc[cc_candidate_list]

        return true_passing_test, cc_candidate, true_passing_test_feature, cc_candidate_feature

    def get_TP_when_not_find_cce(self, data_df):
        failing_df = data_df[data_df["error"] == 1]
        passing_df = data_df[data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i], self.cita):
                    CCE.append(i)
        new_data_df = passing_df[CCE]
        sum_df = new_data_df.sum(axis=1)
        cc_candidate_list = list(sum_df[sum_df > 0].index)
        true_passing_list = list(sum_df[sum_df == 0].index)
        return data_df.iloc[true_passing_list, :].astype('float32'), data_df.iloc[cc_candidate_list, :].astype(
            'float32')
