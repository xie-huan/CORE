
from cc.cc_baselines.BaseCCPipeline import BaseCCPipeline
from cc.core import run
from cc.triplet_cc_identify.PassingTestsHandler import PassingTestsHandler
from utils.task_util import task_complete


class FeatureIdentification(BaseCCPipeline):

    def __init__(self, project_dir, configs, cita, way):
        super().__init__(project_dir, configs, way)
        self.cita = cita

    def _find_cc_index(self):
        self._find_CCE()
        if len(self.CCE) == 0:
            return
        self.CCE.append("error")
        new_data_df = self.data_df[self.CCE]
        self.true_passing_tests, self.cc_candidates = PassingTestsHandler.get_TP_by_Tech_1(new_data_df,self.cita)
        if self.cc_candidates is None or len(self.true_passing_tests) == 0:
            return
        self.cc_index[list(self.cc_candidates.index)] = True


    def _is_CCE(self, fail_data, pass_data):
        fT = self.getfT(fail_data)
        pT = self.getpT(pass_data)
        if ((fT == 1.0) and (pT < self.cita)):
            return True
        else:
            return False

    def _find_CCE(self):
        data = self.dataloader
        data_df = data.data_df
        failing_df = data_df[data_df["error"] == 1]
        passing_df = data_df[data_df["error"] == 0]
        CCE = []
        for i in failing_df.columns:
            if i != "error":
                if self._is_CCE(failing_df[i], passing_df[i]):
                    CCE.append(i)
        # print(CCE)
        # cct=[]
        self.CCE = CCE

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

def main():
    program_list = [
        "Chart",
        "Closure-2023-12-6-1",
        "Lang",
        "Math",
        "Mockito",
        "Time"
    ]
    run(program_list, "Chart", 1, FeatureIdentification, "2022-7-30-Feature", 1)


if __name__ == "__main__":
    main()
    task_complete("Triplet CC end")

    # configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    # sys.argv = os.path.basename(__file__)
    # ccpl = FeatureIdentification(project_dir, configs, 1, "FeatureIdentification")
    # ccpl.find_cc_index()
    # ccpl.evaluation()
    # ccpl.calRes()
