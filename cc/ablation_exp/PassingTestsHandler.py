class PassingTestsHandler:
    def __init__(self):
        pass

    @staticmethod
    def get_passing_tests(data_df):
        return data_df[data_df["error"] == 0]
