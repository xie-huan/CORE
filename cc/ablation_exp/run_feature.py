from cc.ablation_exp.FeatureIdentification import FeatureIdentification
from cc.core import run


def main():
    program_list = [
        "Chart",
        "Closure-2023-12-6-1",
        "Lang",
        "Math",
        "Mockito",
        "Time"
    ]
    name = "2023-6-12-step-1"
    run(program_list, "Chart", 1, FeatureIdentification, name, 0.9)



if __name__ == "__main__":
    main()
    # task_complete("Triplet CC end")