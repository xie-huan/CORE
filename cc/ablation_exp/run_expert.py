from cc.core import run
from cc.expert_feature_cc_identify.EFCIdentifyWithoutExpertFeature import EFCIdentifyWithoutExpertFeature
from cc.expert_feature_cc_identify.ExpertFeatureCombinedIdentify import ExpertFeatureCombinedIdentify


def main():
    # program_list = [
    #     "Chart",
        # "Lang",
        # "Math",
        # "Mockito",
        # "Time"
    # ]
    # program_list = ["Chart"]
    program_list = ["Chart", "Lang", "Math", "Mockito", "Time"]

    # run(program_list, "Chart", 0, TripletCCIdentify, name, 0.9)
    arg_dict = {
        # "cce_threshold":[i/100 for i in range(60, 91, 5)],
        "cce_threshold": 1,
        "select_ratio": [i / 100 for i in range(5, 31, 5)],
        "sus_threshold": [i / 100 for i in range(50, 91, 5)]
    }
    # name = "2022-11-10-triplet-trim-" + str(true_ratio) + "-" + str(select_ratio) + "-70-"
    # multi_process_run(program_list, TripletCCIdentify, name, arg_dict)
    name = "EFC_2023-12-28_100"
    run(program_list, "Chart", 1, ExpertFeatureCombinedIdentify, name, arg_dict)


if __name__ == "__main__":
    main()
    # task_complete("Triplet CC end")
    # configs = {'-d': 'd4j', '-p': 'Chart', '-i': '0', '-m': 'dstar', '-e': 'origin'}
    # sys.argv = ["CCGroundTruthPipeline.py"]
    # ccpl = TripletCCIdentify(project_dir, configs, 1, "2022-8-7-Triplet-Lang")
    # ccpl.find_cc_index()
