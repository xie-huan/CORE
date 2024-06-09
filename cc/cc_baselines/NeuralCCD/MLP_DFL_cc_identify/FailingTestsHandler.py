import numpy as np


class FailingTestsHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_failing_tests(data_df):
        return data_df[data_df["error"] == 1]

    @staticmethod
    def get_a_random_failing_test(data_df):
        failing_df = FailingTestsHandler.get_failing_tests(data_df)
        failing_num = len(failing_df)
        if failing_num == 1:
            return failing_df
        else:
            n = np.random.randint(failing_num)
            a = failing_df.iloc[n].astype('float32')
            return a